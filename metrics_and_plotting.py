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
from prepare_data import file_naming, normalize_layerwise, denormalize_data
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis_fn import get_divergence, get_vorticity, calculate_correlation, get_all_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_colorbar(im, ax, label=None):
    """
    Add a colorbar to a plot.
    Args:
        im: The image to which the colorbar applies
        ax: The axis on which to add the colorbar
        label: Optional label for the colorbar
    Returns:
        The colorbar object
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar

def plot_predictions(int_gt, vel_gt, vel_pred, path_save, filename, plot_intensity=True,  arrows=False):
    """
    Plot ground truth and predicted velocity fields along with intensity maps.

    Args:   
        int_gt: Ground truth intensity map (tensor)
        vel_gt: Ground truth velocity field (2-channel tensor)
        vel_pred: Predicted velocity field (2-channel tensor)
        path_save: Directory to save the plots
        filename: Base name for the saved plot files
        plot_intensity: Boolean indicating whether to plot intensity maps
        arrows: Boolean indicating whether to overlay arrows on velocity fields
    """

    matplotlib.use('agg')
    plt.rcParams['font.size'] = 25

    if plot_intensity:
        nrows = 3
        figsize = (12, 18)
    else:
        nrows = 2
        figsize = (12, 12)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, figsize=figsize)

    intensity_0 = int_gt[int_gt.shape[0]//2 - 1, :, :].cpu().numpy()
    intensity_1 = int_gt[int_gt.shape[0]//2, :, :].cpu().numpy()
    
    vel_0 = vel_gt[0, :, :].cpu().numpy()
    vel_1 = vel_gt[1, :, :].cpu().numpy()

    vel_0_pred = vel_pred[0, :, :].cpu().numpy()
    vel_1_pred = vel_pred[1, :, :].cpu().numpy()

    vmin = -3 * torch.std(vel_gt) / 1e5
    vmax = -1 * vmin
    extent = [0, vel_0.shape[1] * 0.016, 0, vel_0.shape[0] * 0.016]
    if arrows:
        H, W = vel_0.shape
        X, Y = np.meshgrid(np.arange(H) * 0.016, np.arange(W) * 0.016, indexing='ij')
        step = 17
        #quiver_scale = 20  
        quiver_scale = 40
        quiver_width = 0.02
        quiver_alpha = 0.8


    idx = 0
    if plot_intensity:
        for j, intensity in enumerate([intensity_0, intensity_1]):
            im = ax[idx][j].imshow(intensity.T, cmap='magma', origin='lower', extent=extent)
            ax[idx][j].set_title(f'Intensity - channel {j}')
            divider = make_axes_locatable(ax[idx][j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
        idx += 1

    # Ground truth
    im = ax[idx][0].imshow(vel_0.T / 1e5, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][0].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0.T[::step, ::step], vel_1.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha
        )
    title_padding = 27
    ax[idx][0].set_title('vx - ground truth', pad=title_padding)
    divider = make_axes_locatable(ax[idx][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    im = ax[idx][1].imshow(vel_1.T / 1e5, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][1].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0.T[::step, ::step], vel_1.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha
        )
    ax[idx][1].set_title('vy - ground truth', pad=title_padding)
    divider = make_axes_locatable(ax[idx][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    idx += 1

    # Predicted
    im = ax[idx][0].imshow(vel_0_pred.T / 1e5, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][0].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0_pred.T[::step, ::step], vel_1_pred.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha
        )
    ax[idx][0].set_title('vx - predicted', pad=title_padding)
    divider = make_axes_locatable(ax[idx][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    im = ax[idx][1].imshow(vel_1_pred.T / 1e5, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][1].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0_pred.T[::step, ::step], vel_1_pred.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha
        )
    ax[idx][1].set_title('vy - predicted', pad=title_padding)
    divider = make_axes_locatable(ax[idx][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if arrows:
        plt.savefig(path_save + filename + '_arrows.png', bbox_inches='tight')
    else:
        plt.savefig(path_save + filename + '.png', bbox_inches='tight')
    plt.close()

def plot_test_full_map(deepvel_object, params_model,  dataset_path, test_save_path, name, zoomed_in_size = None, plot_intensity = True, n_input_channels = 2, arrows = False, hybrid1=False, hybrid2=False, return_metrics = False):
    """
    Plot predictions on a full map and save the results.
    Args:
        deepvel_object: Instance of the DeepVel model
        params_model: Model parameters
        dataset_path: Path where the dataset is located
        test_save_path: Directory to save the test plots
        name: Base name for the saved plot files
        zoomed_in_size: Optional tuple specifying the size to zoom in on (height, width)
        plot_intensity: Boolean indicating whether to plot intensity maps
        n_input_channels: Number of input channels in the intensity map (2, 4, or 6)
        arrows: Boolean indicating whether to overlay arrows on velocity fields
        hybrid1: Boolean indicating if using hybrid model type 1 (requires I, B and vz)
        hybrid2: Boolean indicating if using hybrid model type 2 (requires B and vz)
        return_metrics: Boolean indicating whether to return calculated metrics
    """
    
    intensity_full_map = np.load(dataset_path + f"intensities_{n_input_channels}.npy")
    velocity_full_map = np.load(dataset_path + f"velocities_{n_input_channels}.npy")

    if hybrid1 or hybrid2:
        B = np.load(dataset_path + f"B_{n_input_channels}.npy")
        vz = np.load(dataset_path + f"vz_{n_input_channels}.npy")

    if zoomed_in_size is not None:
        _, h_full, w_full = intensity_full_map.shape
        h_zoomed_in, w_zoomed_in = zoomed_in_size
        h1 = (h_full//2 - int(h_zoomed_in / 2)) 
        h2 = (h_full//2 + int(h_zoomed_in / 2))
        w1 = (w_full//2 - int(w_zoomed_in / 2)) 
        w2 = (w_full//2 + int(w_zoomed_in / 2))

        intensity_to_plot = intensity_full_map[:, h1:h2, w1:w2]
        B = B[:, h1:h2, w1:w2] if hybrid1 or hybrid2 else None
        vz = vz[:, h1:h2, w1:w2] if hybrid1 or hybrid2 else None
        velocity_to_plot = velocity_full_map[:, h1:h2, w1:w2]
    else:
        intensity_to_plot = intensity_full_map
        velocity_to_plot = velocity_full_map

    intensity_to_plot = torch.from_numpy(intensity_to_plot.astype(np.float32))
    velocity_to_plot = torch.from_numpy(velocity_to_plot.astype(np.float32))

    if hybrid1:
        velocity_pred = deepvel_object.predict(intensity_to_plot, B, vz, params_model)
    elif hybrid2:
        velocity_pred = deepvel_object.predict(B, vz, params_model)
    else:   
        velocity_pred = deepvel_object.predict(intensity_to_plot, params_model)

    plot_predictions(
        intensity_to_plot, velocity_to_plot, velocity_pred,
        test_save_path, name,
        plot_intensity=plot_intensity,
        arrows=arrows
    )
    plot_scatter_plot(velocity_to_plot, velocity_pred, test_save_path, 'scatter_'+name)

    divergence_orig = get_divergence(velocity_to_plot[0].cpu().numpy(), velocity_to_plot[1].cpu().numpy())
    divergence_pred = get_divergence(velocity_pred[0].cpu().numpy(), velocity_pred[1].cpu().numpy())
    vorticity_orig = get_vorticity(velocity_to_plot[0].cpu().numpy(), velocity_to_plot[1].cpu().numpy())
    vorticity_pred = get_vorticity(velocity_pred[0].cpu().numpy(), velocity_pred[1].cpu().numpy())

    mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(velocity_to_plot, velocity_pred).values()
    mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
    mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

    if return_metrics:
        return {
            "mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "slope_vx": slope_x, "slope_vy": slope_y, 
            "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
            "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v
        }


def plot_scatter_plot (original_velocity, predicted_velocity, path_save, filename, plotting_dim = None, zoom_out = 0):
    """
    Create scatter plots comparing original and predicted velocity components.
    Args:
        original_velocity: Original velocity tensor (2-channel)
        predicted_velocity: Predicted velocity tensor (2-channel)
        path_save: Directory to save the scatter plot
        filename: Base name for the saved scatter plot file
        plotting_dim: Optional tuple specifying the dimensions to zoom in on (height, width)
        zoom_out: Value to expand the axes limits for better visualization
    """

    if plotting_dim != None:
        _, h_full, w_full = original_velocity.shape
        h_zoomed_in, w_zoomed_in = plotting_dim

        # new coordinates:
        h1 = (h_full//2 - int(h_zoomed_in / 2)) 
        h2 = (h_full//2 + int(h_zoomed_in / 2))
        w1 = (w_full//2 - int(w_zoomed_in / 2)) 
        w2 = (w_full//2 + int(w_zoomed_in / 2))

        original_velocity_plot = original_velocity[:, h1:h2, w1:w2]
        predicted_velocity_plot = predicted_velocity[:, h1:h2, w1:w2]
    else:
        original_velocity_plot = original_velocity
        predicted_velocity_plot = predicted_velocity
    

    original_velocity_plot = original_velocity_plot.cpu().numpy()
    predicted_velocity_plot = predicted_velocity_plot.cpu().numpy()


    matplotlib.use('agg')
    plt.rcParams['font.size'] = 13
    plt.clf()
    plt.figure(figsize=[15,15])
    fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True)

    im = ax[0].scatter(original_velocity_plot[0].flatten(), predicted_velocity_plot[0].flatten(), alpha = 0.05, linewidths = 0.7)
    min0 = min(original_velocity_plot[0].min(), predicted_velocity_plot[0].min()) - zoom_out
    max0 = max(original_velocity_plot[0].max(), predicted_velocity_plot[0].max()) + zoom_out
    ax[0].plot([min0, max0], [min0, max0], color = 'red') 
    ax[0].set_title('Vx - original vs predicted')
    ax[0].set_xlabel('Original data')
    ax[0].set_ylabel('Predicted data')

    im = ax[1].scatter(original_velocity_plot[1].flatten(), predicted_velocity_plot[1].flatten(), alpha = 0.05,linewidths = 0.7)
    min1 = min(original_velocity_plot[1].min(), predicted_velocity_plot[1].min()) - zoom_out
    max1 = max(original_velocity_plot[1].max(), predicted_velocity_plot[1].max()) + zoom_out
    ax[1].plot([min1, max1], [min1, max1], color = 'red') 
    ax[1].set_title('Vy - original vs predicted')
    ax[1].set_xlabel('Original data')
    ax[1].set_ylabel('Predicted data')

    plt.savefig(path_save+filename + '.png')
    plt.close()


def plot_prediction_and_scatter_full_map_vertical (deepvel_object, params_model, dataset_path, test_save_path, name, zoomed_in_size = None, plot_intensity = True, n_input_channels = 2, scatter_dim = None, zoom_out = 0, write_metrics = False, denormalized = False, return_metrics = False, hybrid1=False, hybrid2=False, title = None):
    """
    Plot predictions and scatter plots for a full map in a vertical layout.
    Args:
        deepvel_object: Instance of the DeepVel model
        params_model: Model parameters
        dataset_path: Path where the dataset is located
        test_save_path: Directory to save the test plots
        name: Base name for the saved plot files
        zoomed_in_size: Optional tuple specifying the size to zoom in on (height, width)
        plot_intensity: Boolean indicating whether to plot intensity maps
        n_input_channels: Number of input channels in the intensity map (2, 4, or 6)
        scatter_dim: Optional tuple specifying the dimensions to zoom in on for scatter plot (height, width)
        zoom_out: Value to expand the axes limits for better visualization in scatter plot
        write_metrics: Boolean indicating whether to write metrics on the plot
        denormalized: Boolean indicating whether to denormalize velocity data before plotting
        return_metrics: Boolean indicating whether to return calculated metrics
        hybrid1: Boolean indicating if using hybrid model type 1 (requires I, B and vz)
        hybrid2: Boolean indicating if using hybrid model type 2 (requires B and vz)
        title: Optional title for the plot
    """
    plt.rcParams['font.size'] = 45
    intensity_full_map = np.load(dataset_path + f"intensities_{n_input_channels}.npy")
    velocity_full_map = np.load(dataset_path + f"velocities_{n_input_channels}.npy")

    if hybrid1 or hybrid2:
        B = np.load(dataset_path + f"B_{n_input_channels}.npy")
        vz = np.load(dataset_path + f"vz_{n_input_channels}.npy")

    extent = [0, velocity_full_map.shape[1]*0.016, 0, velocity_full_map.shape[0]*0.016]

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

    if hybrid1:
        velocity_pred = deepvel_object.predict(intensity_to_plot, B, vz, params_model)

    elif hybrid2:
        velocity_pred = deepvel_object.predict(B, vz, params_model)
        
    else:
        velocity_pred = deepvel_object.predict(intensity_to_plot, params_model)
        
    mse_l = nn.functional.mse_loss(velocity_pred.to(device), velocity_to_plot.to(device))
    rmse_l = torch.sqrt(mse_l)

    matplotlib.use('agg')
    plt.figure()

    nrows = 4 if plot_intensity else 3

    if denormalized: figsize = (25, 40)
    else: figsize=(25, 35)

    fig, ax = plt.subplots(nrows = nrows, ncols = 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.4)


    if plot_intensity == True:
        intensity_0 = intensity_to_plot[n_input_channels/2 -1, :, :].cpu().numpy()
        intensity_1 = intensity_to_plot[n_input_channels/2, :, :].cpu().numpy()


    vel_0 = velocity_to_plot[0, :, :].cpu().numpy()
    vel_1 = velocity_to_plot[1, :, :].cpu().numpy()

    vel_0_pred = velocity_pred[0, :, :].cpu().numpy()
    vel_1_pred = velocity_pred[1, :, :].cpu().numpy()

    idx = 0

    #TODO intensity denormalization option

    if plot_intensity == True:
        middle_channel = int(n_input_channels/2)
        im = ax[idx][0].imshow(intensity_0.T, cmap='magma', origin = 'lower', extent = extent)
        ax[idx][0].set_title(f'Intensity - channel {middle_channel - 1}', pad = 20)
        ax[idx][0].set_xlabel('x [Mm]')
        ax[idx][0].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][0])

        im = ax[idx][1].imshow(intensity_1.T, cmap='magma', origin = 'lower', extent = extent)
        ax[idx][1].set_title(f'Intensity - channel {middle_channel}', pad = 20)
        ax[idx][1].set_xlabel('x [Mm]')
        ax[idx][1].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][1])

        idx += 1

    mean_vel = 1360.96728515625
    std_vel = 230181.609375
    vmin = -3 * std_vel/1e5
    vmax = -1 * vmin

    if denormalized:
        vel_0 = denormalize_data(vel_0, mean_vel, std_vel)/1e5
        vel_1 = denormalize_data(vel_1, mean_vel, std_vel)/1e5
        vel_0_pred = denormalize_data(vel_0_pred, mean_vel, std_vel)/1e5
        vel_1_pred = denormalize_data(vel_1_pred, mean_vel, std_vel)/1e5
        vmin = -3 * std_vel/1e5
        vmax = -1 * vmin

    im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent)
    ax[idx][0].set_title('vx - ground truth', pad = 20)
    ax[idx][0].set_xlabel('x [Mm]')
    ax[idx][0].set_ylabel('y [Mm]')
    cbar = add_colorbar(im, ax[idx][0])

    ax[idx][1].imshow(vel_1.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent)
    ax[idx][1].set_title('vy - ground truth', pad = 20)
    ax[idx][1].set_xlabel('x [Mm]')
    ax[idx][1].set_ylabel('y [Mm]')
    cbar = add_colorbar(im, ax[idx][1])

    idx +=1

    ax[idx][0].imshow(vel_0_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent)
    ax[idx][0].set_title('vx - predicted', pad = 20)
    ax[idx][0].set_xlabel('x [Mm]')
    ax[idx][0].set_ylabel('y [Mm]')
    cbar = add_colorbar(im, ax[idx][0])

    ax[idx][1].imshow(vel_1_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent)
    ax[idx][1].set_title('vy - predicted', pad = 20)
    ax[idx][1].set_xlabel('x [Mm]')
    ax[idx][1].set_ylabel('y [Mm]')
    cbar = add_colorbar(im, ax[idx][1])

    idx +=1

    vmin2 = -3 * np.std(vel_0)/1e5
    vmax2 = -1 * vmin2

    im = ax[idx][0].scatter(vel_0.flatten(), vel_0_pred.flatten(), alpha = 0.05, linewidths = 0.7)
    min0 = min(vel_0.min(), vel_0_pred.min()) - zoom_out
    max0 = max(vel_0.max(), vel_0_pred.max()) + zoom_out
    ax[idx][0].plot([min0, max0], [min0, max0], color = 'red') 
    ax[idx][0].set_title('Vx - original vs predicted', pad = 20)
    ax[idx][0].set_xlabel('Original data')
    ax[idx][0].set_ylabel('Predicted data')

    im = ax[idx][1].scatter(vel_1.flatten(), vel_1_pred.flatten(), alpha = 0.05, linewidths = 0.7)
    min1 = min(vel_1.min(), vel_1_pred.min()) - zoom_out
    max1 = max(vel_1.max(), vel_1_pred.max()) + zoom_out
    ax[idx][1].plot([min1, max1], [min1, max1], color = 'red') 
    ax[idx][1].set_title('Vy - original vs predicted', pad = 20)
    ax[idx][1].set_xlabel('Original data')
    ax[idx][1].set_ylabel('Predicted data')
    
    divergence_orig = get_divergence(vel_0, vel_1)
    divergence_pred = get_divergence(vel_0_pred, vel_1_pred)
    vorticity_orig = get_vorticity(vel_0, vel_1)
    vorticity_pred = get_vorticity(vel_0_pred, vel_1_pred)

    mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(velocity_to_plot, velocity_pred).values()
    mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
    mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

    if write_metrics: fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}', 
         ha='center', fontsize=16)
    
    else: 
        print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}')
        print(f'Divergence - MSE Loss: {mse_d:.4f}, Root MSE Loss: {rmse_d:.4f}, Pearson: {pearson_d:.3f}, Slope: {slope_d:.3f}')
        print(f'Vorticity - MSE Loss: {mse_v:.4f}, Root MSE Loss: {rmse_v:.4f}, Pearson: {pearson_v:.3f}, Slope: {slope_v:.3f}')
    
    if title is not None:
        plt.suptitle(title, fontsize=70, y=0.98)

    if test_save_path is not None:
        plt.savefig(test_save_path + 'full_analysis_' + name + '.png')
    plt.close()

    if return_metrics:
        return {"mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "slope_vx": slope_x, "slope_vy": slope_y, 
                "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
                "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v}
