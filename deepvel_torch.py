import numpy as np
import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
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
from torcheval.metrics.functional import binary_accuracy
import time
import matplotlib
from prepare_data import normalize_layerwise
from metrics_and_plotting import plot_predictions, plot_test_full_map, plot_scatter_plot, calculate_correlation


class dataset_deepVel(Dataset): 
    """
    for Stefani's project
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


    def __getitem__(self, idx):

        if self.test == False:
            image = np.load(self.intensities_dir + "/"+ self.intensities[idx])
            vel = np.load(self.velocities_dir + "/"+ self.velocities[idx])
            #print("image train: ", self.intensities[idx])
            #print("velocity train: ", self.velocities[idx])

            if self.velocities[idx][11:16] != self.intensities[idx][12:17]:
                raise ValueError("Input - label data not corresponding: ", self.intensities[idx] + " "+ self.velocities[idx])

        else:
            image = np.load(self.intensities_dir + "/"+ self.intensities[self.validation_idx[idx]])
            vel = np.load(self.velocities_dir + "/"+ self.velocities[self.validation_idx[idx]])
            #print("image val: ", self.validation_idx[idx])
            #print("velocity val: ", self.validation_idx[idx])  
            if self.velocities[self.validation_idx[idx]][11:16] != self.intensities[self.validation_idx[idx]][12:17]:
                raise ValueError("Input - label data not corresponding: ", self.intensities[self.validation_idx[idx] ]+ " "+ self.velocities[self.validation_idx[idx]])
        
        image = normalize_layerwise(image)
        vel = normalize_layerwise(vel)

        image = torch.from_numpy(image.astype(np.float32))
        vel = torch.from_numpy(vel.astype(np.float32))

        return image, vel
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                        #nn.Tanh() #because data is around -1 - +1?
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
    def __init__(self, in_channels=2, out_channels=2, n_filters=32, blocks=20): #changed out_channels
        super(DeepVel_net, self).__init__()
        #parameters
        self.n_filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        #Layers
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, self.n_filters, kernel_size = 3, stride=1, padding=1),
                                       nn.BatchNorm2d(self.n_filters),
                                       nn.ReLU()
                                       #nn.Tanh()
                                       )
        
        self.residual = self.make_Reslayer(self.n_filters, self.blocks)

        self.conv2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters, kernel_size = 3, stride=1, padding=1), nn.BatchNorm2d(self.n_filters))
        self.conv3 = nn.Conv2d(self.n_filters, self.out_channels, kernel_size = 1, stride=1, padding=0)

    def make_Reslayer(self, out_channels, num_blocks, stride=1):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.n_filters, out_channels, stride))
                self.n_filters = out_channels
            return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        res = out
        out = self.residual(out)
        out = self.conv2(out)
        out += res
        out = self.conv3(out)
        return out


class DeepVel_run(object):
    def __init__(self, root = None):
 
        self.root = root
        self.n_filters = 32     
        self.batch_size = 64
        self.n_conv_layers = 20 #Number of residual blocks
        self.in_channels = 2
        self.out_channels = 2
        #self.lr = 1e-3
        self.lr = 1e-4

        self.model = DeepVel_net(in_channels=self.in_channels, out_channels=self.out_channels, n_filters=self.n_filters, blocks=self.n_conv_layers).to(device)

        if root:

            self.dataset = dataset_deepVel('dataset_cropped_17k')
            self.train_len = int(0.8*len(self.dataset))
            self.test_len = len(self.dataset) - self.train_len

            self.trainset, self.testset = random_split(self.dataset, [self.train_len, self.test_len])

            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
            self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle = False, num_workers = 0)

            self.summary = summary(self.model, input_size = (self.in_channels, 64, 64), batch_size = self.batch_size) #I will cut out images 64 x 64

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    


    def train(self, epochs):

        """
        Used to train DeepVel

        Parameters
        ----------
        epochs : int
            Refer to the number of epochs in the training
        
        """

        min_loss = torch.tensor(float('inf'))
        loss_list = []
        save_losses = []

        #scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.96)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                outputs = self.model(x.to(device))
                #if (epoch == 0 and batch_i == 0):
                #    print(x.shape, y.shape)
                #    plt.figure(figsize=[10,4])
                #    plt.subplot(121)
                #    plt.imshow(x[0,0].T, origin='lower', vmin=-2,vmax=2)
                #    plt.subplot(122)
                #    plt.imshow(x[0,1].T, origin='lower', vmin=-2,vmax=2)
                #    plt.savefig("test_within_training.png")
                
                loss = self.criterion(outputs, y.to(device))
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
                val_loss = self.criterion(output, y.to(device))

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
                torch.save(self.model.state_dict(), 'network/DeepVel_torch_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))

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
        with open('network/deepvel_torch_train_params_{}.npy'.format(date_time), 'wb') as f:
            np.save(f, dict)
            np.save(f, save_losses)
        
    def predict(self, x, saved_model):

        """
        Class used to predict using a deepvel saved model

        Parameters
        ----------
        x : Tensor object 
            Continumm image with a size torch.Size([2, H, W])

        saved_model: torch model .pt
            trained model

        norm_file : numpy array .npz 
            saved values used for normalization then are use to retrive the physical quantities of the dataset

        """

        model_weights = torch.load(saved_model, map_location=torch.device(device))
        self.model.load_state_dict(model_weights)
        self.model.eval()

        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x.astype(np.float32))
        
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = normalize_layerwise(x)
        start = time.time()
    
        with torch.no_grad():    
            output = self.model(x.to(device))  

        end = time.time()
        print("Prediction took {0} seconds...".format(end-start))

        #output = normalization(output, norm_file)
        output = output.squeeze(0)
        output = normalize_layerwise(output)
        
        return output
       
if (__name__ == '__main__'):

    main_root = r"/home/xenoss/data/kecman_project/DeepVel_3D_velocity"
    deepvel_v1 = DeepVel_run(main_root)

    #"We used batches of 32 samples and trained the network for 30 epochs, where an epoch is finished once all training samples have been used." 
    #deepvel_v1.train(300)
    
    params_model = main_root+r'/network/expo_scheduler_Adam_cropped_17k_lr1e-2_600epochs/DeepVel_torch_epoch_65_0.22539.pt'
    test_save_path = "test_results/trained_on_17k_300_expo_scheduler_Adam/"
    
    test_indices = [0, 50, 100, 150, 200, 250]
    for ti in test_indices:
        db_check = deepvel_v1.testset[ti]
        image, vel = db_check
        vel = normalize_layerwise(vel)
        image = normalize_layerwise(image)

        vel_pred = deepvel_v1.predict(image, params_model)
        mse_l = nn.functional.mse_loss(vel_pred.to(device), vel.to(device))
        print("MSE loss: ", mse_l)

        print(calculate_correlation(vel, vel_pred))
        #plot_predictions(image, vel, vel_pred, test_save_path, "test_"+str(ti)) 
        #plot_scatter_plot(vel, vel_pred, test_save_path, "test_scatter_"+str(ti))

    
        
    
    #### TESTING OF THE WHOLE MAP ####

    #plot_test_full_map(8, deepvel_v1, params_model, test_save_path, name = "not_zoomed_in_full_map")
