import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import random
from torchsummary import summary
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
import time
import matplotlib
#from prepare_data import normalize_layerwise
from metrics_and_plotting import plot_predictions, plot_test_full_map, plot_scatter_plot, calculate_correlation, plot_prediction_and_scatter_full_map_vertical
from analysis_fn import div_vor_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class dataset_deepVel(Dataset): 
    """
    DeepVel dataset class.

    Custom Dataset based on Dataloaders https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """ 
    def __init__(self, dataset_path, test_idx = None, test = False):
        self.intensities_dir = os.path.join(dataset_path, "inputs")
        self.velocities_dir = os.path.join(dataset_path, "labels")

        intensities = os.listdir(self.intensities_dir)
        velocities = os.listdir(self.velocities_dir)

        intensities.sort()
        velocities.sort()

        self.intensities = intensities
        self.velocities = velocities

        self.n_train_datapoints = len(self.intensities)
        
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

        intensity_index = self.intensities[idx].replace('intensity_', '')
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

        I = np.load(self.intensities_dir + "/"+ self.intensities[idx])
        vel = np.load(self.velocities_dir + "/"+ self.velocities[idx])

        self.check_indices(idx, self.test)

        I = torch.from_numpy(I.astype(np.float32))
        vel = torch.from_numpy(vel.astype(np.float32))

        return I, vel
    
class ResidualBlock(nn.Module):
    """
    Class for the residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
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
    def __init__(self, in_channels=6, out_channels=2, n_filters=32, blocks=20): 
        super(DeepVel_net, self).__init__()

        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.n_filters),
                                       nn.ReLU()
                                       )
        
        self.residual = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))
        self.conv3 = nn.Conv2d(self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

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
        return out


class DeepVel_run(object):
    """
    Class for running the DeepVel model.
    """
    def __init__(self, batch, dataset_path, network_path, in_channels = 2, root = None,):
 
        self.root = root
        self.n_filters = 32     
        self.batch_size = batch
        self.n_conv_layers = 20 
        self.in_channels = in_channels
        self.out_channels = 2
        self.lr = 1e-4
        self.network_path = network_path

        self.model = DeepVel_net(in_channels=self.in_channels, out_channels=self.out_channels, n_filters=self.n_filters, blocks=self.n_conv_layers).to(device)

        if root:

            self.dataset = dataset_deepVel(dataset_path, test_idx = None, test = False)
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len

            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])

            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 0)

            self.summary = summary(self.model, input_size = (self.in_channels, 128, 128), batch_size = self.batch_size) #TODO fix hardcoding 128,128

        #self.criterion = nn.MSELoss()
        #self.criterion = div_vor_loss
        self.criterion = lambda pred, gt: div_vor_loss(pred, gt, alpha1=1, alpha2=1.384507e-5*1e3, alpha3=1.434438e-05*1e3, normalize=True)
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

        # Paths for saving MSEs
        path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_8/"
        mse1_path = os.path.join(path, "mse1.npy")
        mse2_path = os.path.join(path, "mse2.npy")
        mse3_path = os.path.join(path, "mse3.npy")

        # Load existing values if files exist, else start new lists
        if os.path.exists(mse1_path):
            mse1_list = list(np.load(mse1_path))
        else:
            mse1_list = []
        
        if os.path.exists(mse2_path):
            mse2_list = list(np.load(mse2_path))
        else:
            mse2_list = []

        if os.path.exists(mse3_path):
            mse3_list = list(np.load(mse3_path))
        else:
            mse3_list = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            epoch_mse1 = []
            epoch_mse2 = []
            epoch_mse3 = []
            for batch_i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(x.to(device))

                loss_tuple = self.criterion(outputs, y.to(device))
                if isinstance(loss_tuple, tuple):
                    loss, mse1, mse2, mse3 = loss_tuple
                    epoch_mse1.append(mse1.item())
                    epoch_mse2.append(mse2.item())
                    epoch_mse3.append(mse3.item())
                else:
                    loss = loss_tuple
                
                loss.backward()

                print("MSE1: ", mse1.item(), " MSE2: ", mse2.item(), " MSE3: ", mse3.item())

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

            # Save average MSEs for this epoch
            if epoch_mse1 and epoch_mse2 and epoch_mse3:
                mse1_list.append(np.mean(epoch_mse1))
                mse2_list.append(np.mean(epoch_mse2))
                mse3_list.append(np.mean(epoch_mse3))
                np.save(mse1_path, mse1_list)
                np.save(mse2_path, mse2_list)
                np.save(mse3_path, mse3_list)

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
                torch.save(self.model.state_dict(), self.network_path + '/DeepVel_torch_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
            
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
        self.model.load_state_dict(model_weights)
        self.model.eval()

        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x.astype(np.float32))
        
        if x.dim() == 3:
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

    deepvel_net = DeepVel_run(root = main_root, in_channels=4, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/normalized/timestep_{4}/cropped/data_{128}x{128}', network_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_8/checkpoints/")
    deepvel_net.train(120)

    ###### NOTE training 5x5 experiments #####

    # temporal_range = [2, 4, 6, 8, 10]
    # spatial_range = [32, 48, 64, 96, 128]

    # for t in temporal_range:
    #     for s in spatial_range:
    #         print(f"Training for timestep {t} and spatial size {s} x {s}")
    #         # name = os.listdir(main_root + f'datasets_5x5_experiments/normalized/timestep_{t}/cropped/data_{s}x{s}/inputs/')[0]
    #         # sample = np.load(main_root + f'datasets_5x5_experiments/normalized/timestep_{t}/cropped/data_{s}x{s}/inputs/' + name)
    #         # print("Sample shape: ", sample.shape)
    #         deepvel_net = DeepVel_run(root = main_root, in_channels=t, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/normalized/timestep_{t}/cropped/data_{s}x{s}', network_path = main_root + f'models_5x5_norm/timestep_{t}/{s}x{s}/')
    #         deepvel_net.train(80)
            
       
    #### NOTE TESTING OF THE WHOLE MAP EXAMPLE ####

    # deepvel_net = DeepVel_run(root = main_root, in_channels=6, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/timestep_{6}/cropped/data_{128}x{128}', network_path = main_root + f'models_5x5/timestep_{6}/{128}x{128}/')
    # params_model = '/dat/xenoss/models_5x5/timestep_6/128x128/DeepVel_torch_epoch_79_0.14405.pt'
    # #plot_test_full_map(8, deepvel_net, params_model, '/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment25_test/', name = "experiment25_not_zoomed_in_full_map")
    # plot_prediction_and_scatter_full_map_vertical(8, deepvel_net, params_model, '/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment25_test/', name = "experiment25_not_zoomed_in_full_map", n_input_channels=6, write_metrics=True)

  