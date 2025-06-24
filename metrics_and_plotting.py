import os
import shutil
import muram as muram
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_predictions(int_gt, vel_gt, vel_pred, path_save, filename):

    matplotlib.use('agg')
    plt.figure(figsize = (100, 100))
    fig, ax = plt.subplots(nrows = 3, ncols = 2, sharex = True, sharey = True)

    intensity_0 = int_gt[0, :, :].cpu().numpy()
    intensity_1 = int_gt[1, :, :].cpu().numpy()

    vel_0 = vel_gt[0, :, :].cpu().numpy()
    vel_1 = vel_gt[1, :, :].cpu().numpy()

    vel_0_pred = vel_pred[0, :, :].cpu().numpy()
    vel_1_pred = vel_pred[1, :, :].cpu().numpy()

    vmin = -3 * torch.std(vel_gt)
    vmax = -1 * vmin

    im = ax[0][0].imshow(intensity_0, cmap='magma')
    ax[0][0].set_title('Intensity - channel 0')
    fig.colorbar(im, ax = ax[0][0])


    im = ax[0][1].imshow(intensity_1, cmap='magma')
    ax[0][1].set_title('Intensity - channel 1')
    fig.colorbar(im, ax = ax[0][1])

    im = ax[1][0].imshow(vel_0, cmap='bwr', vmin = vmin, vmax = vmax)
    ax[1][0].set_title('vx - ground truth')
    fig.colorbar(im, ax = ax[1][0])

    ax[1][1].imshow(vel_1, cmap='bwr', vmin = vmin, vmax = vmax)
    ax[1][1].set_title('vy - ground truth')
    fig.colorbar(im, ax = ax[1][1])

    ax[2][0].imshow(vel_0_pred, cmap='bwr', vmin = vmin, vmax = vmax)
    ax[2][0].set_title('vx - predicted')
    fig.colorbar(im, ax = ax[2][0])

    ax[2][1].imshow(vel_1_pred, cmap='bwr', vmin = vmin, vmax = vmax)
    ax[2][1].set_title('vy - predicted')
    fig.colorbar(im, ax = ax[2][1])

    plt.savefig(path_save+filename + '.png')
    plt.close()

def plot_test_full_map(full_map_idx, deepvel_object, params_model, test_save_path, name, zoomed_in_size = None):

    map_name = file_naming("intensities", full_map_idx)
    
    intensity_full_map = np.load('main_dataset_sorted/inputs/' + map_name +'.npy')
    velocity_full_map = np.load('main_dataset_sorted/labels/' + map_name.replace('intensities', 'velocities') +'.npy')

    intensity_full_map = normalize_layerwise(intensity_full_map)
    velocity_full_map = normalize_layerwise(velocity_full_map)

    if zoomed_in_size != None:
        #plot the central part zoomed in to that size
        print(intensity_full_map.shape)
        _, h_full, w_full = intensity_full_map.shape
        h_zoomed_in, w_zoomed_in = zoomed_in_size

        # new coordinates:
        h1 = (h_full//2 - int(h_zoomed_in / 2)) 
        h2 = (h_full//2 + int(h_zoomed_in / 2))
        w1 = (w_full//2 - int(w_zoomed_in / 2)) 
        w2 = (w_full//2 + int(w_zoomed_in / 2))

        intensity_to_plot = intensity_full_map[:, h1:h2, w1:w2]
        velocity_to_plot = velocity_full_map[:, h1:h2, w1:w2]

    else:
        intensity_to_plot = intensity_full_map
        velocity_to_plot = velocity_full_map

    intensity_to_plot = torch.from_numpy(intensity_to_plot.astype(np.float32))
    velocity_to_plot = torch.from_numpy(velocity_to_plot.astype(np.float32))

    velocity_pred = deepvel_object.predict(intensity_to_plot, params_model)
    mse_l = nn.functional.mse_loss(velocity_pred.to(device), velocity_to_plot.to(device))

    print("MSE loss full map: ", mse_l)

    plot_predictions(intensity_to_plot, velocity_to_plot, velocity_pred, test_save_path, name)

def plot_scatter_plot (original_velocity, predicted_velocity, path_save, filename):

    original_velocity = original_velocity.cpu().numpy()
    predicted_velocity = predicted_velocity.cpu().numpy()

    matplotlib.use('agg')

    plt.clf()
    plt.figure(figsize=[15,8])
    fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True)

    im = ax[0].scatter(original_velocity[0].flatten(), predicted_velocity[0].flatten())
    ax[0].set_title('Vx - original vs predicted')
    ax[0].set_xlabel('Original data')
    ax[0].set_ylabel('Predicted data')

    im = ax[1].scatter(original_velocity[1].flatten(), predicted_velocity[1].flatten())
    ax[1].set_title('Vy - original vs predicted')
    ax[1].set_xlabel('Original data')
    ax[1].set_ylabel('Predicted data')

    plt.savefig(path_save+filename + '.png')
    plt.close()

def calculate_correlation (original, predicted):

    '''
    Pearson correlation coefficient - statistic:

    Ranges from -1 to 1.
    +1: perfect positive linear correlation
    0: no linear correlation
    -1: perfect negative linear correlation

    pvalue: 
    A p-value close to 0 means the observed correlation is highly statistically significant.
    '''
    original = original.cpu().numpy()
    predicted = predicted.cpu().numpy()

    pr_0 = stats.pearsonr(original[0, :, :].flatten(), predicted[0, :, :].flatten())
    pr_1 = stats.pearsonr(original[1, :, :].flatten(), predicted[1, :, :].flatten())

    return [pr_0, pr_1]
