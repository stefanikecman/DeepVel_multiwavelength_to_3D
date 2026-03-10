import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
# from torchsummary import summary
from torchinfo import summary
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dataset_deepVel(Dataset): 
    """
    DeepVel dataset class.

    Custom Dataset based on Dataloaders https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """ 
    def __init__(self, dataset_path, tau, test_idx = None, test = False, stokes_profile = "I", out_channels=None):

        self.stokes_profile = stokes_profile
        self.out_channels = out_channels
        self.intensities_dir = os.path.join(dataset_path, f"inputs/stokes_{self.stokes_profile}")
        print("Loading velocities from tau: ", tau)
        self.velocities_dir = os.path.join(dataset_path, "labels/tau_" + tau)

        intensities = os.listdir(self.intensities_dir)
        velocities = os.listdir(self.velocities_dir)

        intensities.sort()
        velocities.sort()

        if test == False:
            for i in range(len(intensities)):
                intensity_index = intensities[i].replace(f'stokes_{self.stokes_profile}_', '')
                velocity_index = velocities[i].replace('velocities_', '')

                if velocity_index != intensity_index:
                    raise ValueError("Input (I) - label data not corresponding: ", intensities[i] + " "+ velocities[i])
                
            self.intensities = [torch.from_numpy(np.load(os.path.join(self.intensities_dir, f)).astype(np.float32)) for f in intensities]
            self.velocities = [torch.from_numpy(np.load(os.path.join(self.velocities_dir, f)).astype(np.float32)) for f in velocities]
            
            if self.out_channels == 1:
                self.velocities = [vel[2,:,:].unsqueeze(0) for vel in self.velocities]

        # self.intensities = intensities
        # self.velocities = velocities

        self.n_train_datapoints = len(intensities)
        
        self.test = test
        self.test_idx = test_idx

    def __len__(self):

        if self.test and self.test_idx:
            return self.n_test_datapoints
            
        else:
            return self.n_train_datapoints

    def check_indices (self, idx, test=False):
        """
        Check if the input intensity and velocity files correspond to each other.
        Args:
            idx: Index of the data point to check
            test: Boolean indicating if in test mode
        Raises:
            ValueError: If the input and label data do not correspond
        """
        if test == True:
            idx = self.validation_idx[idx]

        # intensity_index = self.intensities[idx].replace('intensity_', '')
        intensity_index = self.intensities[idx].replace(f'stokes_{self.stokes_profile}_', '')
        velocity_index = self.velocities[idx].replace('velocities_', '')

        if velocity_index != intensity_index:
                raise ValueError("Input (I) - label data not corresponding: ", self.intensities[idx] + " "+ self.velocities[idx])

    def __getitem__(self, idx):
        """
        Get a data point from the dataset.
        Args:
            idx: Index of the data point to retrieve
        Returns:
            Tuple of (intensity, velocity) tensors
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if self.test:
            idx = self.validation_idx[idx]

        # I = np.load(os.path.join(self.intensities_dir, self.intensities[idx]))
        # I = torch.from_numpy(I.astype(np.float32))  

        # if self.out_channels == 1:
        #     vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))[2,:,:]
        #     vel = torch.from_numpy(vel.astype(np.float32)).unsqueeze(0)
        # else:
        #     vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))
        #     vel = torch.from_numpy(vel.astype(np.float32))

        # self.check_indices(idx, self.test)

        I = self.intensities[idx]
        vel = self.velocities[idx]

        return I, vel
    
class ResidualBlock(nn.Module):
    """
    Class for the residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU(inplace=True)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm3d(out_channels))
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
class DeepVel_net(nn.Module):
    """
    Model definition 
    """
    def __init__(self, in_channels, out_channels, n_filters=32, blocks=20):
        super(DeepVel_net, self).__init__()

        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        
        self.conv1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm3d(self.n_filters), nn.ReLU(inplace=True))
        # TODO increase the kernel size to 5 or 6
        # Add avg pooling or max pooling around 3, 5 to shrink the wavelengths after every convolutional layer
        # decrease the number of filters to 
        # change the kernel and stride to be different in wav and spatial dimensions
        #in the beginning have bigger kernels and then decrease them. not go above 10 and below 3

        self.residual = self.make_Reslayer(self.n_filters, self.blocks)
        self.conv2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm3d(self.n_filters))
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv3 = nn.Conv3d(self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

    def make_Reslayer(self, out_channels, num_blocks, stride=1):
            """
            Create a sequence of residual blocks.
            Args:
                out_channels: Number of output channels for each block
                num_blocks: Number of residual blocks to create
                stride: Stride for the first block (default is 1)
            Returns:
                A sequential container of residual blocks
            """
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.n_filters, out_channels, stride))
                self.n_filters = out_channels
            return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        out = self.conv1(x)
        res = out
        out = self.residual(out)
        out = self.conv2(out)
        out += res        
        out = self.conv3(out)
        out = self.adapt_pool(out)
        out = out.squeeze(2)
        return out

class DeepVel_net2(nn.Module):
    """
    Model definition. Modified original DeepVel_net:
    - increased kernel size in the first convolutional layer to capture more information from the input data (max 10, min 3)
    - decreased kernel size in the second convolutional layer to capture more local information after the residual blocks
    - decreased the number of filters to reduce the number of parameters
    - added Max pooling in the wavelength domain
    - avoid the adaptive pooling and do max pooling bit by bit to reduce the wavelengths and then pool them to one
    """
    def __init__(self, in_channels, out_channels, n_filters=16, blocks=20):
        super(DeepVel_net2, self).__init__()

        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        
        self.conv1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = (10, 5, 5), stride=1, padding=(2,2,2)), nn.BatchNorm3d(self.n_filters), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 1, 1), stride=(5, 1, 1))
        self.residual = self.make_Reslayer(self.n_filters, self.blocks)
        self.conv2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = (5, 3, 3), stride=1, padding=(2,1,1)), nn.BatchNorm3d(self.n_filters))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.pool_res = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.conv3 = nn.Conv3d(self.n_filters, self.out_channels, kernel_size = (1, 1, 1), stride=1, padding=0)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.adapt_pool = nn.AdaptiveMaxPool3d((1, None, None))

    def make_Reslayer(self, out_channels, num_blocks, stride=1):
            """
            Create a sequence of residual blocks.
            Args:
                out_channels: Number of output channels for each block
                num_blocks: Number of residual blocks to create
                stride: Stride for the first block (default is 1)
            Returns:
                A sequential container of residual blocks
            """
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.n_filters, out_channels, stride))
                self.n_filters = out_channels
            return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        out = self.conv1(x)
        out = self.pool1(out)
        res = out
        out = self.residual(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out += self.pool_res(res)  #to match depth after pool2        
        out = self.conv3(out)
        out = self.max_pool(out)
        out = self.adapt_pool(out)
        out = out.squeeze(2)
        return out


class DeepVel_run(object):
    """
    Class for running the DeepVel model.
    """
    def __init__(self, batch, dataset_path, network_path, tau, in_shape = (2, 43, 64, 64), out_shape = (3, 64, 64), root = None, stokes_profile="I", test=False):
 
        self.root = root
        self.n_filters = 16  
        self.batch_size = batch
        self.n_conv_layers = 20 
        self.lr = 1e-4
        self.network_path = network_path
        self.in_channels, self.n_wavelengths, self.height, self.width = in_shape
        self.out_channels = out_shape[0]
        self.stokes_profile = stokes_profile

        self.model = DeepVel_net(in_channels=self.in_channels, out_channels=self.out_channels, n_filters=self.n_filters, blocks=self.n_conv_layers)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(device)

        if root:

            self.dataset = dataset_deepVel(dataset_path, tau = tau, test_idx = None, test = test, stokes_profile=self.stokes_profile, out_channels=self.out_channels)
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len
            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])
            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 0)
            self.summary = summary(self.model, input_data = (torch.randn(self.batch_size, self.in_channels, self.n_wavelengths, self.height, self.width)), device=device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)    

    def train(self, epochs):

        """
        Used to train DeepVel

        Args:
            epochs : number of epochs in the training

        """

        min_loss = torch.tensor(float('inf'))
        loss_list = []
        save_losses = []

        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.96)

        with open(os.path.join(self.network_path.replace("checkpoints/", ""), "architecture.txt"), "w") as f:
            print("Writing architecture to file...", os.path.join(self.network_path.replace("checkpoints/", ""), "architecture.txt"))
            f.write(str(self.summary))

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(x.to(device))

                loss_tuple = self.criterion(outputs, y.to(device))
                if isinstance(loss_tuple, tuple):
                    loss, mse1, mse2, mse3 = loss_tuple
                else:
                    loss = loss_tuple
                
                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()
                loss_list.append(loss.cpu().detach().numpy())
        
                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    epochs,
                    batch_i,
                    len(self.trainloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                    )
                )
            scheduler.step()

            val_loss_list = []
            val_acc_list = []

            self.model.eval()
            for batch_i, (x, y) in enumerate(self.testloader):
                with torch.no_grad():    
                    output = self.model(x.to(device))  
                val_loss_tuple = self.criterion(output, y.to(device))
                if isinstance(val_loss_tuple, tuple):
                    val_loss = val_loss_tuple[0]
                else:
                    val_loss = val_loss_tuple

                val_loss_list.append(val_loss.cpu().detach().numpy())
        

            print(' Epoch {} - Loss : {:.5f} - Validation loss : {:.5f}'.format(epoch, 
                                                                                np.mean(loss_list), 
                                                                                np.mean(val_loss_list)))
            save_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])
        
            compare_loss = np.mean(val_loss_list)
            is_best = compare_loss < min_loss
            print(min_loss, compare_loss)
            if is_best == True:
                print("Best_model")      
                min_loss = min(compare_loss, min_loss)
                save_path = os.path.join(self.network_path, 'DeepVel_torch_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
                
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                    'best_loss': min_loss,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss_history': loss_list,
                    'valid_loss_history': val_loss_list,
                }
                
                # model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save(checkpoint, save_path)
            
        print('Finished Training')

        dt = datetime.now()
        date_time = dt.strftime("%m_%d_%Y_%H_%M_%S")
        dict = OrderedDict()
        dict['root'] = self.root
        dict['dataset_lenght'] = len(self.trainset)+len(self.trainset)
        dict['channels'] = self.n_filters
        dict['Number_epochs'] = epochs
        dict['Bach_size'] = self.batch_size
        dict['Optimizer_lr'] = self.lr

        with open(self.network_path + '/deepvel_torch_train_params_{}.npy'.format(date_time), 'wb') as f:
        
            np.save(f, dict)
            np.save(f, save_losses)
        
    def predict(self, x, saved_model):

        """
        Class used to predict using a deepvel saved model

        Args:
            x : Tensor object 
                Continumm image with a size torch.Size([2, H, W])

        saved_model: torch model .pt
            trained model

        norm_file : numpy array .npz 
            saved values used for normalization then are use to retrive the physical quantities of the dataset

        Returns:
            output : Tensor object
                Predicted velocity field with a size torch.Size([2, H, W])

        """
        model_weights = torch.load(saved_model, map_location=torch.device(device))
        
        if isinstance(model_weights, dict) and 'state_dict' in model_weights:
            model_weights = model_weights['state_dict']
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_weights)
        else:
            self.model.load_state_dict(model_weights)

        self.model.eval()

        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x.astype(np.float32))
        
        if x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(0)

        elif x.dim() == 4:
            x = x.unsqueeze(0)

        #x = normalize_layerwise(x)
        start = time.time()
    
        with torch.no_grad():    
            output = self.model(x.to(device))  

        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        #output = normalization(output, norm_file)
        output = output.squeeze(0)
        #output = normalize_layerwise(output)
        
        return output
       
if (__name__ == '__main__'):

    main_root = "/dat/xenoss/"
    ###### NOTE best version ######
    input_shape = (2, 43, 64, 64)  # (channels, wavelengths, height, width)
    #output_shape = (3, 64, 64)    # (velocity components, height, width)
    output_shape = (1, 64, 64)    # (vz, height, width)
    
    # taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
    taus = ["1e-1", "1e-2", "1e-2", "1e-3", "1e-4"] 
    stokes_profiles = ["V", "I", "V", "V", "V"]
    dataset_path = '/dat/xenoss/thesis/data/v1/dataset/train/'

    # for tau in taus:
    #     for stokes_profile in stokes_profiles:
    #         deepvel_net = DeepVel_run(root = main_root, tau = tau, in_shape = input_shape, out_shape=output_shape, batch = 8, dataset_path = dataset_path, network_path = f"/scratch/xenoss/stokes_{stokes_profile.lower()}_vz/tau_{tau}/checkpoints/", stokes_profile=stokes_profile)
    #         deepvel_net.train(200)  

    for i in range(len(taus)):
        tau = taus[i]
        stokes_profile = stokes_profiles[i]
        deepvel_net = DeepVel_run(root = main_root, tau = tau, in_shape = input_shape, out_shape=output_shape, batch = 8, dataset_path = dataset_path, network_path = f"/scratch/xenoss/stokes_{stokes_profile.lower()}_vz/tau_{tau}/checkpoints/", stokes_profile=stokes_profile)
        deepvel_net.train(200)