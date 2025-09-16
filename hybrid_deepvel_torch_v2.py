import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import random
#from torchsummary import summary
import torch.optim as optim
import sys
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt
from torcheval.metrics.functional import binary_accuracy
import time
import matplotlib
#from prepare_data import normalize_layerwise
from metrics_and_plotting import plot_predictions, plot_test_full_map, plot_scatter_plot, calculate_correlation, plot_prediction_and_scatter_full_map_vertical
from analysis_fn import div_vor_loss


class dataset_deepVel(Dataset): 
    """
    Dataset class for DeepVel, modified to take two inputs: B and vz.
    """ 
    def __init__(self, dataset_path, in_channels, test_idx = None, test = False):
        self.in_channels = in_channels
        self.magnetic_fields_dir = os.path.join(dataset_path, "B")
        self.vz_dir = os.path.join(dataset_path, "vz")
        self.velocities_dir = os.path.join(dataset_path, "labels")

        velocities = os.listdir(self.velocities_dir)
        magnetic_fields = os.listdir(self.magnetic_fields_dir)
        vz = os.listdir(self.vz_dir)

        velocities.sort()
        magnetic_fields.sort()
        vz.sort()

        self.velocities = velocities
        self.magnetic_fields = magnetic_fields
        self.vz = vz

        self.n_train_datapoints = len(self.vz)
        
        #figure out test indices
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
        mag_index = self.magnetic_fields[idx].replace('B_', '')
        vz_index = self.vz[idx].replace('vz_', '')

        if mag_index != velocity_index:
                raise ValueError("Input (B) - label data not corresponding: ", self.magnetic_fields[idx] + " "+ self.velocities[idx])
        if vz_index != velocity_index:
                raise ValueError("Input (vz) - label data not corresponding: ", self.vz[idx] + " "+ self.velocities[idx])

    def __getitem__(self, idx):
        """
        Get a data point from the dataset.
        Args:
            idx: Index of the data point to retrieve
        Returns:
            Tuple of (magnetic field, vz, velocity) tensors
        Raises:
            ValueError: If the input and label data do not correspond
        """

        if self.test:
            idx = self.validation_idx[idx]

        B = np.load(self.magnetic_fields_dir + "/"+ self.magnetic_fields[idx])
        vz = np.load(self.vz_dir + "/"+ self.vz[idx])
        vel = np.load(self.velocities_dir + "/"+ self.velocities[idx])

        self.check_indices(idx, self.test)

        return B, vz, vel

class ResidualBlock(nn.Module):
    """
    Residual block definition
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
        Model definition for DeepVel, modified to take magnetic field and vz as inputs.
    """
    def __init__(self, in_channels=6, out_channels=2, n_filters=32, blocks=20): #changed out_channels
        super(DeepVel_net, self).__init__()
        #parameters
        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        #Layers
        self.conv_B_1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.n_filters),
                                       nn.ReLU()
                                       )
        self.conv_vz_1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.n_filters),
                                        nn.ReLU()
                                        )
        
        self.residual2 = self.make_Reslayer(self.n_filters, self.blocks)
        self.residual3 = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv_B_2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))
        self.conv_vz_2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))

        self.conv3 = nn.Conv2d(2*self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

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

    def forward(self, B, vz):
        """
        Forward pass through the network.
        """
        out_B = self.conv_B_1(B)
        res_B = out_B

        out_vz = self.conv_vz_1(vz)
        res_vz = out_vz

        out_B = self.residual2(out_B)
        out_B = self.conv_B_2(out_B)
        out_B += res_B

        out_vz = self.residual3(out_vz)
        out_vz = self.conv_vz_2(out_vz)
        out_vz += res_vz

        feat_ensemble = torch.cat((out_B, out_vz), dim=1) #myb dim 0
        out = self.conv3(feat_ensemble)

        return out


class DeepVel_run(object):
    """
    Class for running the DeepVel model, modified to take two inputs: B and vz.
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
        self.dataset = dataset_deepVel(dataset_path, in_channels=self.in_channels, test_idx=None, test=False)
        self.model = DeepVel_net(in_channels=self.in_channels, out_channels=self.out_channels, n_filters=self.n_filters, blocks=self.n_conv_layers).to(device)

        if root:

            self.dataset = dataset_deepVel(dataset_path, in_channels=self.in_channels, test_idx=None, test=False)
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len

            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])

            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 0)

            #self.summary = summary(self.model, input_size = (self.in_channels, 128, 128), batch_size = self.batch_size) #this does not handle multiple inputs

        self.criterion = nn.MSELoss()
        # self.criterion = div_vor_loss
        # self.criterion = lambda pred, gt: div_vor_loss(pred, gt, alpha1=1.0, alpha2=1e8, alpha3=1e8)
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

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (B, vz, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(B.to(device), vz.to(device))

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
            for batch_i, (B, vz, y) in enumerate(self.testloader):
                with torch.no_grad():    
                    output = self.model(B.to(device), vz.to(device))

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
    
    def to_tensor(self, x):
            """
            Convert input to tensor and add batch dimension if necessary.
            """
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return x
         
    def predict(self, B, vz, saved_model):
        """
        Predict using a DeepVel saved model with two inputs: B, vz.

        Args:
            B : Tensor or np.ndarray
                Magnetic field input, shape [C, H, W] or [1, C, H, W]
            vz : Tensor or np.ndarray
                vz input, shape [C, H, W] or [1, C, H, W]
            saved_model: str
                Path to trained model (.pt file)
        Returns:
            output : Tensor
                Predicted velocity field, shape [out_channels, H, W]
        Raises: 
            ValueError: If input dimensions are incorrect
        """

        model_weights = torch.load(saved_model, map_location=torch.device(device))
        self.model.load_state_dict(model_weights)
        self.model.eval()

        B = self.to_tensor(B)
        vz = self.to_tensor(vz)

        start = time.time()
        with torch.no_grad():
            output = self.model(B.to(device), vz.to(device))
        end = time.time()
        print("Prediction took {0} seconds...".format(end - start))

        output = output.squeeze(0)
        return output
       
if (__name__ == '__main__'):

    main_root = "/dat/xenoss/"
    ###### NOTE best version ######

    deepvel_net = DeepVel_run(root = main_root, in_channels=4, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/normalized/timestep_{4}/cropped/data_{128}x{128}', network_path = "hybrid_model_v2/mse_loss/checkpoints/")
    deepvel_net.train(80)

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


