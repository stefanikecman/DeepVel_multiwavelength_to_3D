import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
#from torchsummary import summary
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import time
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dataset_deepVel(Dataset): 
    """
    Dataset class for DeepVel, modified to take two inputs: Stokes I and Stokes V.
    """ 
    def __init__(self, dataset_path, test_idx = None, test = False, tau = None, out_channels = None):
        self.stokes_I_dir = os.path.join(dataset_path, "inputs/stokes_I") 
        self.stokes_V_dir = os.path.join(dataset_path, "inputs/stokes_V")
        self.velocities_dir = os.path.join(dataset_path, "labels") if tau is None else os.path.join(dataset_path, f"labels/tau_{tau}")
        self.out_channels = out_channels

        velocities = os.listdir(self.velocities_dir)
        stokes_I = os.listdir(self.stokes_I_dir)
        stokes_V = os.listdir(self.stokes_V_dir)

        velocities.sort()
        stokes_I.sort()
        stokes_V.sort()

        stokes_I = stokes_I#[:31360] #only for experiment 1
        stokes_V = stokes_V#[:31360] #only for experiment 1
        velocities = velocities#[:31360] #only for experiment 1
 
        if test==False:
            for v, I, V in zip(velocities, stokes_I, stokes_V):
                velocity_index = v.replace('velocities_', '')
                stokes_I_index = I.replace('stokes_I_', '')
                stokes_V_index = V.replace('stokes_V_', '')

                if stokes_I_index != velocity_index:
                    raise ValueError("Input (stokes I) - label data not corresponding: ", I + " "+ v)
                if stokes_V_index != velocity_index:
                    raise ValueError("Input (stokes V) - label data not corresponding: ", V + " "+ v)

            self.velocities = [torch.from_numpy(np.load(os.path.join(self.velocities_dir, f)).astype(np.float32)) for f in velocities]
            self.stokes_I = []
            for f in stokes_I:
                i = torch.from_numpy(np.load(os.path.join(self.stokes_I_dir, f)).astype(np.float32))
                if i.dim() == 3:
                    i = i.unsqueeze(0)
                self.stokes_I.append(i)

            self.stokes_V = []
            for f in stokes_V:
                i = torch.from_numpy(np.load(os.path.join(self.stokes_V_dir, f)).astype(np.float32))
                if i.dim() == 3:
                    i = i.unsqueeze(0)
                self.stokes_V.append(i)
            
            # self.stokes_I = [torch.from_numpy(np.load(os.path.join(self.stokes_I_dir, f)).astype(np.float32)) for f in stokes_I]
            # self.stokes_V = [torch.from_numpy(np.load(os.path.join(self.stokes_V_dir, f)).astype(np.float32)) for f in stokes_V]

            if self.out_channels == 1:
                self.velocities = [vel[2,:,:].unsqueeze(0) for vel in self.velocities]

            elif self.out_channels == 2:
                self.velocities = [vel[:2,:,:].unsqueeze(0) for vel in self.velocities]

            print("Loaded {} velocity files, {} stokes I files, {} stokes V files.".format(len(self.velocities), len(self.stokes_I), len(self.stokes_V)))

        self.n_train_datapoints = len(stokes_V)
        
        self.test = test
        self.test_idx = test_idx

    def __len__(self):

        if self.test and self.test_idx:
            return self.n_test_datapoints
            
        else:
            return self.n_train_datapoints

    def check_indices (self, idx, test=False):
        """
        Check if the input magnetic field and vz files correspond to the velocity files.
        Args:
            idx: Index of the data point to check
            test: Boolean indicating if in test mode
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if test == True:
            idx = self.validation_idx[idx]

        velocity_index = self.velocities[idx].replace('velocities_', '')
        stokes_I_index = self.stokes_I[idx].replace('stokes_I_', '')
        stokes_V_index = self.stokes_V[idx].replace('stokes_V_', '')

        if stokes_I_index != velocity_index:
                raise ValueError("Input (stokes I) - label data not corresponding: ", self.stokes_I[idx] + " "+ self.velocities[idx])
        if stokes_V_index != velocity_index:
                raise ValueError("Input (stokes V) - label data not corresponding: ", self.stokes_V[idx] + " "+ self.velocities[idx])

    def __getitem__(self, idx):
        """
        Get a data point from the dataset.
        Args:
            idx: Index of the data point to retrieve
        Returns:
            Tuple of (Stokes I, Stokes V, velocity) tensors
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if self.test:
            idx = self.validation_idx[idx]

        # stokes_I = np.load(os.path.join(self.stokes_I_dir, self.stokes_I[idx]))
        # stokes_V = np.load(os.path.join(self.stokes_V_dir, self.stokes_V[idx]))
        
        # stokes_I = torch.from_numpy(stokes_I.astype(np.float32))
        # stokes_V = torch.from_numpy(stokes_V.astype(np.float32))

        # if self.out_channels == 1:
        #     vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))[2,:,:]
        #     vel = torch.from_numpy(vel.astype(np.float32)).unsqueeze(0)
        # else: 
        #     vel = np.load(os.path.join(self.velocities_dir, self.velocities[idx]))
        #     vel = torch.from_numpy(vel.astype(np.float32))       

        # self.check_indices(idx, self.test)

        stokes_I = self.stokes_I[idx]
        stokes_V = self.stokes_V[idx]
        vel = self.velocities[idx]

        return stokes_I, stokes_V, vel

class ResidualBlock(nn.Module):
    """
    Residual block definition
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU()
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
        Model definition for DeepVel, modified to take magnetic field and vz as inputs.
    """
    def __init__(self, in_channels, out_channels, n_filters=32, blocks=20):
        super(DeepVel_net, self).__init__()

        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks

        self.conv_I_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                       nn.BatchNorm3d(self.n_filters),
                                       nn.ReLU()
                                       )
        self.conv_V_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                        nn.BatchNorm3d(self.n_filters),
                                        nn.ReLU()
                                        )
        
        self.residual2 = self.make_Reslayer(self.n_filters, self.blocks)
        self.residual3 = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv_I_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm3d(self.n_filters))
        self.conv_V_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm3d(self.n_filters))
        # self.adapt_pool_I = nn.AdaptiveAvgPool3d((1, None, None))
        # self.adapt_pool_V = nn.AdaptiveAvgPool3d((1, None, None))
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv3 = nn.Conv3d(2*self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

    def make_Reslayer(self, out_channels, num_blocks, stride=1):
            """
            Create a sequence of residual blocks.
            Args:
                out_channels: Number of output channels for each block
                num_blocks: Number of residual blocks to create
                stride: Stride for the first block
            Returns:
                nn.Sequential containing the residual blocks
            """
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.n_filters, out_channels, stride))
                self.n_filters = out_channels
            return nn.Sequential(*layers)

    def forward(self, I, V):
        """
        Forward pass through the network.
        """
        out_I = self.conv_I_1(I)
        res_I = out_I

        out_V = self.conv_V_1(V)
        res_V = out_V

        out_I = self.residual2(out_I)
        out_I = self.conv_I_2(out_I)
        out_I += res_I
        

        out_V = self.residual3(out_V)
        out_V = self.conv_V_2(out_V)
        out_V += res_V

        feat_ensemble = torch.cat((out_I, out_V), dim=1)

        out = self.conv3(feat_ensemble)

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
        

        self.conv_I_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = (10, 5, 5), stride=1, padding=(2,2,2)),
                                       nn.BatchNorm3d(self.n_filters),
                                       nn.ReLU(inplace=True)
                                       )
        self.pool_I_1 = nn.MaxPool3d(kernel_size=(5, 1, 1), stride=(5, 1, 1))

        self.conv_V_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = (10, 5, 5), stride=1, padding=(2,2,2)),
                                        nn.BatchNorm3d(self.n_filters),
                                        nn.ReLU(inplace=True)
                                        )
        self.pool_V_1 = nn.MaxPool3d(kernel_size=(5, 1, 1), stride=(5, 1, 1))

        self.residual2 = self.make_Reslayer(self.n_filters, self.blocks)
        self.residual3 = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv_I_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = (5, 3, 3), stride=1, padding=(2,1,1)), 
                                      nn.BatchNorm3d(self.n_filters))
        self.pool_I_2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        self.pool_res_I = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        
        self.conv_V_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = (5, 3, 3), stride=1, padding=(2,1,1)), 
                                      nn.BatchNorm3d(self.n_filters))
        
        self.pool_V_2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))        
        self.pool_res_V = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))

        self.conv3 = nn.Conv3d(2*self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)
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

    def forward(self, I, V):
        """
        Forward pass through the network.
        """  

        out_I = self.conv_I_1(I)
        out_I = self.pool_I_1(out_I)
        res_I = out_I

        out_V = self.conv_V_1(V)
        out_V = self.pool_V_1(out_V)
        res_V = out_V

        out_I = self.residual2(out_I)
        out_I = self.conv_I_2(out_I)
        out_I = self.pool_I_2(out_I)
        out_I += self.pool_res_I(res_I)
        

        out_V = self.residual3(out_V)
        out_V = self.conv_V_2(out_V)
        out_V = self.pool_V_2(out_V)
        out_V += self.pool_res_V(res_V)

        feat_ensemble = torch.cat((out_I, out_V), dim=1)

        out = self.conv3(feat_ensemble)
        out = self.max_pool(out)
        out = self.adapt_pool(out)
        out = out.squeeze(2)

        return out

class DeepVel_net3(nn.Module):
    """
    Model definition. Modified original DeepVel_net:
    - increased kernel size in the first convolutional layer to capture more information from the input data (max 10, min 3)
    - decreased kernel size in the second convolutional layer to capture more local information after the residual blocks
    - decreased the number of filters to reduce the number of parameters
    - added Max pooling in the wavelength domain
    - avoid the adaptive pooling and do max pooling bit by bit to reduce the wavelengths and then pool them to one
    """
    def __init__(self, in_channels, out_channels, n_filters=16, blocks=20):
        super(DeepVel_net3, self).__init__()

        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        

        self.conv_I_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = (25, 5, 5), stride=1, padding=(12,2,2)),
                                       nn.BatchNorm3d(self.n_filters), nn.ReLU(inplace=True))
        self.pool_I_1 = nn.MaxPool3d(kernel_size=(15, 1, 1), stride=(15, 1, 1))

        self.conv_V_1 = nn.Sequential(nn.Conv3d(self.in_channels, self.n_filters, kernel_size = (25, 5, 5), stride=1, padding=(12,2,2)),
                                        nn.BatchNorm3d(self.n_filters), nn.ReLU(inplace=True))
        self.pool_V_1 = nn.MaxPool3d(kernel_size=(15, 1, 1), stride=(15, 1, 1))

        self.residual2 = self.make_Reslayer(self.n_filters, self.blocks)
        self.residual3 = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv_I_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = (15, 3, 3), stride=1, padding=(7,1,1)), 
                                      nn.BatchNorm3d(self.n_filters))
        self.pool_I_2 = nn.MaxPool3d(kernel_size=(9, 1, 1), stride=(9, 1, 1))
        self.pool_res_I = nn.MaxPool3d(kernel_size=(9, 1, 1), stride=(9, 1, 1))
        
        self.conv_V_2 = nn.Sequential(nn.Conv3d(self.n_filters, self.n_filters, kernel_size = (15, 3, 3), stride=1, padding=(7,1,1)), 
                                      nn.BatchNorm3d(self.n_filters))
        
        self.pool_V_2 = nn.MaxPool3d(kernel_size=(9, 1, 1), stride=(9, 1, 1))        
        self.pool_res_V = nn.MaxPool3d(kernel_size=(9, 1, 1), stride=(9, 1, 1))

        self.conv3 = nn.Conv3d(2*self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)
        # self.max_pool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        
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

    def forward(self, I, V):
        """
        Forward pass through the network.
        """  

        out_I = self.conv_I_1(I)
        out_I = self.pool_I_1(out_I)
        res_I = out_I

        out_V = self.conv_V_1(V)
        out_V = self.pool_V_1(out_V)
        res_V = out_V

        out_I = self.residual2(out_I)
        out_I = self.conv_I_2(out_I)
        out_I = self.pool_I_2(out_I)
        out_I += self.pool_res_I(res_I)
        

        out_V = self.residual3(out_V)
        out_V = self.conv_V_2(out_V)
        out_V = self.pool_V_2(out_V)
        out_V += self.pool_res_V(res_V)

        feat_ensemble = torch.cat((out_I, out_V), dim=1)

        out = self.conv3(feat_ensemble)
        # out = self.max_pool(out)
        out = self.adapt_pool(out)
        out = out.squeeze(2)

        return out


class DeepVel_run(object):
    """
    Class for running the DeepVel model, modified to take two inputs: Stokes I and Stokes V.
    """
    def __init__(self, batch, dataset_path, network_path, in_shape = (2, 43, 64, 64), out_shape = (3, 64, 64), root = None, tau = None, test=False):
 
        self.root = root
        self.n_filters = 16
        # self.n_filters = 32 #for v3     
        self.batch_size = batch
        self.n_conv_layers = 20
        self.tau = tau
        self.in_channels, self.n_wavelengths, self.height, self.width = in_shape
        self.out_channels = out_shape[0]
        self.lr = 1e-4
        self.network_path = network_path
        self.model = DeepVel_net2(in_channels=self.in_channels, out_channels=self.out_channels, n_filters=self.n_filters, blocks=self.n_conv_layers).to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        if root:
            self.dataset = dataset_deepVel(dataset_path, test_idx=None, test=test, tau = self.tau, out_channels=self.out_channels)
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len

            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])

            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 4, pin_memory=True)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 4, pin_memory=True)

            self.summary = summary(self.model, input_data = (torch.randn(self.batch_size, self.in_channels, self.n_wavelengths, self.height, self.width), torch.randn(self.batch_size, self.in_channels, self.n_wavelengths, self.height, self.width)), device=device)

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
            for batch_i, (I, V, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(I.to(device), V.to(device))

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
            for batch_i, (I, V, y) in enumerate(self.testloader):
                with torch.no_grad():    
                    output = self.model(I.to(device), V.to(device))

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
        dict['dataset_length'] = len(self.trainset)+len(self.testset)
        dict['channels'] = self.n_filters
        dict['Number_epochs'] = epochs
        dict['Bach_size'] = self.batch_size
        dict['Optimizer_lr'] = self.lr

        with open(os.path.join(self.network_path, 'deepvel_torch_train_params_{}.npy'.format(date_time)), 'wb') as f:
        
            np.save(f, dict)
            np.save(f, save_losses)
    
    def to_tensor(self, x):
            """
            Convert input to tensor and add batch dimension if necessary.
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.dim() == 3:
                x = x.unsqueeze(0).unsqueeze(0)
            if x.dim() == 4:
                x = x.unsqueeze(0)
            return x
         
    def predict(self, I, V, saved_model):
        """
        Predict using a DeepVel saved model with two inputs: I, V.

        Args:
            I : Tensor or np.ndarray
                Stokes I input, shape [C, lambda, H, W] or [1, C, lambda, H, W]
            V : Tensor or np.ndarray
                Stokes V input, shape [C, lambda, H, W] or [1, C, lambda, H, W]
            saved_model: str
                Path to trained model (.pt file)
        Returns:
            output : Tensor
                Predicted velocity field, shape [out_channels, H, W]
        Raises: 
            ValueError: If input dimensions are incorrect
        """

        model_weights = torch.load(saved_model, map_location=torch.device(device))

        if isinstance(model_weights, dict) and 'state_dict' in model_weights:
            model_weights = model_weights['state_dict']
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model_weights)
        else:
            self.model.load_state_dict(model_weights)
        self.model.eval()
        I = self.to_tensor(I)
        V = self.to_tensor(V)
        
        start = time.time()
        with torch.no_grad():
            output = self.model(I.to(device), V.to(device))
        end = time.time()
        print("Prediction took {0} seconds...".format(end - start))

        output = output.squeeze(0)
        return output
       
if (__name__ == '__main__'):

    main_root = "/dat/xenoss/"

    taus = ["1e-1"]
    version = "v4"
    dataset_path = f'/dat/xenoss/thesis/data/{version}/dataset/train/'
    input_shape = (2, 175, 16, 16)  # (channels, wavelengths, height, width)
    output_shape = (1, 16, 16)    # (velocity components, height, width)

    for tau in taus:
        deepvel_net = DeepVel_run(root = main_root, tau = tau, test=False, in_shape = input_shape, out_shape=output_shape, batch = 256, dataset_path = dataset_path, 
                                    network_path = f"/scratch/xenoss/{version}/tau_{tau}/hybrid_vz_model/checkpoints/")
        deepvel_net.train(100)