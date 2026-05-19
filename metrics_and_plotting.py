import glob
import os
import muram as muram
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analysis_fn import get_divergence, get_vorticity, get_all_metrics, denormalize_data
from enum import Enum

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelType(Enum):
    DEEPVEL_I = 1
    DEEPVEL_B = 2
    HYBRID_Bvz = 3
    HYBRID_IBvz = 4
    STOKES_I = 5
    STOKES_V = 6
    STOKES_IV = 7
    DEEPVEL_Vz = 8
    HYBRID_Ivz = 9

def infer_with_sliding_windows(model, params_model, input, window=128, overlap=0):

    if input.ndim == 5:
        n_data, _, _, H, W = np.array(input).shape
        hybrid = True if  n_data== 2 else False
    elif input.ndim == 4:
        _, _, H, W = np.array(input).shape
        hybrid = False
    print("Input shape:", np.array(input).shape)
    
    first = True
    for y in range(0, H, window-overlap):
        for x in range(0, W, window-overlap):

            if hybrid:
                I_patch = input[0, :, :, y:y+window, x:x+window]
                V_patch = input[1, :, :, y:y+window, x:x+window]
                pred = model.predict(I_patch, V_patch, params_model)
                if first:
                    output_channels = pred.shape[0]
                    output = torch.zeros(output_channels, H, W)
                    first = False

            else:
                if input.ndim == 5:
                    input_patch = input[0, :, :, y:y+window, x:x+window]
                elif input.ndim == 4:
                    input_patch = input[:, :, y:y+window, x:x+window]
                pred = model.predict(input_patch, params_model)
                if first:
                    output_channels = pred.shape[0]
                    output = torch.zeros(output_channels, H, W)
                    first = False

            output[:, y:y+window, x:x+window] = pred
    return output

def plot_train_val_losses(tau_levels, ckpts_path, model_type: ModelType, title,save_path, name):
    """
    Method for plot_train_val_losses
    
    Args:
        tau_levels: integers corresponding to tau levels, first one should always be intensity
        ckpts_path: path where the checkpoints are stored
    """
    taus = [str(tau) for tau in tau_levels]
    all_x = []
    all_train = []
    all_val = []
    
    for i, tau in enumerate(tau_levels):
        # print(f"tau = {tau}, model type = {model_type[i]}, ckpts_path = {ckpts_path}")
        if model_type[i] == ModelType.DEEPVEL_I:
            model_glob = f"{ckpts_path}/DeepVel_I/checkpoints/*.npy"
        elif model_type[i] == ModelType.DEEPVEL_B:
            model_glob = f"{ckpts_path}/DeepVel_Bz/tau_{tau}/checkpoints/*.npy"
        elif model_type[i] == ModelType.DEEPVEL_Vz:
            model_glob = f"{ckpts_path}/DeepVel_vz/tau_{tau}/checkpoints/*.npy"
        elif model_type[i] == ModelType.HYBRID_Bvz:
            model_glob = f"{ckpts_path}/Hybrid_B_vz/tau_{tau}/checkpoints/*.npy"
        elif model_type[i] == ModelType.HYBRID_IBvz:
            model_glob = f"{ckpts_path}/Hybrid_I_B_vz/tau_{tau}/checkpoints/*.npy"
        elif model_type[i] == ModelType.HYBRID_Ivz:
            model_glob = f"{ckpts_path}/Hybrid_I_vz/tau_{tau}/checkpoints/*.npy"

        elif model_type[i] == ModelType.STOKES_I:
            model_glob = f"{ckpts_path}/tau_{tau}/stokes_i_model/checkpoints/*.npy"
        elif model_type[i] == ModelType.STOKES_V:
            model_glob = f"{ckpts_path}/tau_{tau}/stokes_v_model/checkpoints/*.npy"
        elif model_type[i] == ModelType.STOKES_IV:
            # model_glob = f"{ckpts_path}/tau_{tau}/hybrid_model/checkpoints/*.npy"
            model_glob = f"{ckpts_path}/tau_{tau}/hybrid_vz/checkpoints/*.npy"

        path = glob.glob(model_glob)[0]
        with open(path, "rb") as f:
            _ = np.load(f, allow_pickle=True).item()
            save_losses = np.load(f, allow_pickle=True)

        save_losses = np.asarray(save_losses, dtype=float)
        x = save_losses[:, 0]
        train_rmse = np.sqrt(save_losses[:, 1])
        val_rmse = np.sqrt(save_losses[:, 2])

        all_x.append(x)
        all_train.append(train_rmse)
        all_val.append(val_rmse)

    plt.figure(figsize=(14, 9))
    plt.rcParams['font.size'] = 14

    cmap = plt.get_cmap('tab10')
    n = len(all_train)

    for i in range(n):
        x = np.asarray(all_x[i])
        train = np.asarray(all_train[i])
        val = np.asarray(all_val[i])
        color = cmap(i % cmap.N)
        if model_type[i] == ModelType.DEEPVEL_I: 
            label = "DeepVel I"
        elif model_type[i] == ModelType.DEEPVEL_B:
            label = f"DeepVel Bz tau={taus[i]}"
        elif model_type[i] == ModelType.DEEPVEL_Vz:
            label = f"DeepVel Vz tau={taus[i]}"
        elif model_type[i] == ModelType.HYBRID_Bvz:
            label = f"Hybrid Bz+Vz tau={taus[i]}"
        elif model_type[i] == ModelType.HYBRID_IBvz:
            label = f"Hybrid I+Bz+Vz tau={taus[i]}"
        elif model_type[i] == ModelType.HYBRID_Ivz:
            label = f"Hybrid I+Vz tau={taus[i]}"
        elif model_type[i] == ModelType.STOKES_I:
            label = f"Stokes I tau={taus[i]}"
        elif model_type[i] == ModelType.STOKES_V:
            label = f"Stokes V tau={taus[i]}"

        elif model_type[i] == ModelType.STOKES_IV:
            label = f"Hybrid Stokes IV tau={taus[i]}"
        plt.plot(x, train, color=color, linestyle='-', linewidth=2, label=f"{label} train")
        plt.plot(x, val,   color=color, linestyle='--', linewidth=1.5, label=f"{label} val")

    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSE)')
    plt.title(title)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    out_path = os.path.join(save_path, name + '.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()

def load_data_based_on_model_type(dataset_path, n_input_channels, model_type: ModelType, tau = None):
    """
    Load all possible input data based on the model type
    """

    inputs = {
        "intensity": None,
        "B": None,
        "vz": None,
        "stokes_I": None,
        "stokes_V": None
    }
    if model_type == ModelType.HYBRID_IBvz or model_type== ModelType.DEEPVEL_I or model_type== ModelType.HYBRID_Ivz:
        if tau is not None:
            inputs["intensity"] = np.load(os.path.join(dataset_path, f"intensities_tau_{tau}.npy"))
        else:
            inputs["intensity"] = np.load(os.path.join(dataset_path, f"intensities_tau_{n_input_channels}.npy"))


    if model_type == ModelType.HYBRID_Bvz or model_type== ModelType.DEEPVEL_B or model_type== ModelType.HYBRID_IBvz:
        if tau is not None:
            inputs["B"] = np.load(os.path.join(dataset_path, f"B_tau_{tau}.npy"))
            
        else:
            inputs["B"] = np.load(os.path.join(dataset_path, f"B_{n_input_channels}.npy"))
            
    if model_type == ModelType.HYBRID_Bvz or model_type== ModelType.HYBRID_IBvz  or model_type== ModelType.DEEPVEL_Vz or model_type== ModelType.HYBRID_Ivz:
        if tau is not None:
            inputs["vz"] = np.load(os.path.join(dataset_path, f"vz_tau_{tau}.npy"))
        else:
            inputs["vz"] = np.load(os.path.join(dataset_path, f"vz_{n_input_channels}.npy"))

    if model_type == ModelType.STOKES_I or model_type == ModelType.STOKES_IV:
        inputs["stokes_I"] = np.load(os.path.join(dataset_path, f"inputs/stokes_I/stokes_I_{n_input_channels}.npy"))

    if model_type == ModelType.STOKES_V or model_type == ModelType.STOKES_IV:
        inputs["stokes_V"] = np.load(os.path.join(dataset_path, f"inputs/stokes_V/stokes_V_{n_input_channels}.npy"))

    return inputs

def to_torch (data):
    if data is not None:
        return torch.from_numpy(data.astype(np.float32))
    else:
        return None
    
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

def load_data_and_get_predictions(deepvel_object, params_model, dataset_path, n_input_channels, n_out_channels, model_type: ModelType, tau = None, zoomed_in_size = None, sliding_windows = False):

    inputs = load_data_based_on_model_type(dataset_path, n_input_channels, model_type, tau)

    intensity, B, vz, stokes_I, stokes_V = inputs["intensity"], inputs["B"], inputs["vz"], inputs["stokes_I"], inputs["stokes_V"]

    if tau is not None:
        print('Loading vel from: ', os.path.join(dataset_path, f"labels/tau_{tau}/velocities_tau_{tau}.npy"))
        velocity_full_map = np.load(os.path.join(dataset_path, f"labels/tau_{tau}/velocities_tau_{tau}.npy"))
    else:
        velocity_full_map = np.load(os.path.join(dataset_path, f"labels/velocities_{n_input_channels}.npy"))

    if n_out_channels == 1:
        velocity_full_map = velocity_full_map[:, 2, :, :] #for vz only inference
    elif n_out_channels == 2:
        velocity_full_map = velocity_full_map[:, :2, :, :] #for vx, vy inference

    if velocity_full_map.ndim == 4:
        velocity_full_map = velocity_full_map[0]

    if zoomed_in_size != None:
        #plot the central part zoomed in to that size
        print(intensity.shape)
        _, h_full, w_full = intensity.shape
        h_zoomed_in, w_zoomed_in = zoomed_in_size

        # new coordinates:
        h1 = (h_full//2 - int(h_zoomed_in / 2)) 
        h2 = (h_full//2 + int(h_zoomed_in / 2))
        w1 = (w_full//2 - int(w_zoomed_in / 2)) 
        w2 = (w_full//2 + int(w_zoomed_in / 2))

        if intensity is not None:
            intensity_to_plot = intensity[:, h1:h2, w1:w2]
        if B is not None:
            B_to_plot = B[:, h1:h2, w1:w2]
        if vz is not None:
            vz_to_plot = vz[:, h1:h2, w1:w2]
        if stokes_I is not None:
            stokes_I_to_plot = stokes_I[:, h1:h2, w1:w2]
            #stokes_I_to_plot = torch.stack((stokes_I_to_plot, stokes_I_to_plot), dim=0) #NOTE just for one test - duplicate the stokes I to have 2 channels for the model
        if stokes_V is not None:
            stokes_V_to_plot = stokes_V[:, h1:h2, w1:w2]
        velocity_to_plot = velocity_full_map[:, h1:h2, w1:w2]

    else:
        intensity_to_plot = intensity
        B_to_plot = B
        vz_to_plot = vz
        stokes_I_to_plot = stokes_I
        stokes_V_to_plot = stokes_V
        velocity_to_plot = velocity_full_map

    intensity_to_plot = to_torch(intensity_to_plot)
    velocity_to_plot = to_torch(velocity_to_plot)
    B_to_plot = to_torch(B_to_plot)
    vz_to_plot = to_torch(vz_to_plot)
    stokes_I_to_plot = to_torch(stokes_I_to_plot)
    stokes_V_to_plot = to_torch(stokes_V_to_plot)
    
    
    

    if model_type == ModelType.HYBRID_IBvz:
        velocity_pred = deepvel_object.predict(intensity_to_plot, B_to_plot, vz_to_plot, params_model)

    elif model_type == ModelType.HYBRID_Ivz:
        velocity_pred = deepvel_object.predict(intensity_to_plot, vz_to_plot, params_model)

    elif model_type == ModelType.HYBRID_Bvz:
        velocity_pred = deepvel_object.predict(B_to_plot, vz_to_plot, params_model)

    elif model_type == ModelType.DEEPVEL_B:
        velocity_pred = deepvel_object.predict(B_to_plot, params_model)

    elif model_type == ModelType.DEEPVEL_I:
        velocity_pred = deepvel_object.predict(intensity_to_plot, params_model)
    
    elif model_type == ModelType.DEEPVEL_Vz:
        velocity_pred = deepvel_object.predict(vz_to_plot, params_model)

    elif model_type == ModelType.STOKES_I:
        stokes_I_to_plot = stokes_I_to_plot.unsqueeze(0) if n_out_channels==1 else stokes_I_to_plot
        if sliding_windows:
            print("Predicting with sliding windows")
            velocity_pred = infer_with_sliding_windows(deepvel_object, params_model, torch.from_numpy(np.array([stokes_I_to_plot])), window=256, overlap=0)
        else:
            velocity_pred = deepvel_object.predict(stokes_I_to_plot, params_model)

    elif model_type == ModelType.STOKES_V:
        stokes_V_to_plot = stokes_V_to_plot.unsqueeze(0) if n_out_channels==1 else stokes_V_to_plot
        if sliding_windows:
            print("Predicting with sliding windows")
            velocity_pred = infer_with_sliding_windows(deepvel_object, params_model, torch.from_numpy(np.array([stokes_V_to_plot])), window=256, overlap=0)
        else:   
            velocity_pred = deepvel_object.predict(stokes_V_to_plot, params_model)
        
    elif model_type == ModelType.STOKES_IV:
        stokes_I_to_plot = stokes_I_to_plot.unsqueeze(0) if n_out_channels==1 else stokes_I_to_plot
        stokes_V_to_plot = stokes_V_to_plot.unsqueeze(0) if n_out_channels==1 else stokes_V_to_plot
        if sliding_windows:
            print("Predicting with sliding windows")
            velocity_pred = infer_with_sliding_windows(deepvel_object, params_model, torch.from_numpy(np.array([stokes_I_to_plot, stokes_V_to_plot])), window=256, overlap=0)
        else:
            velocity_pred = deepvel_object.predict(stokes_I_to_plot, stokes_V_to_plot, params_model)
            print("velocity_pred shape:", velocity_pred.shape)

    return inputs, velocity_to_plot, velocity_pred

def plot_predictions(input_gt, vel_gt, vel_pred, path_save, filename, plot_intensity=True,  arrows=False):
    """
    Plot ground truth and predicted velocity fields along with intensity maps.

    Args:   
        input_gt: Ground truth input map (tensor)
        vel_gt: Ground truth velocity field (2 or 3-channel tensor)
        vel_pred: Predicted velocity field (2 or 3-channel tensor)
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

    if vel_gt.shape[0]==3:
        nrows +=1
        figsize = (12, 24)

    fig, ax = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True, figsize=figsize)

    input_0 = input_gt[input_gt.shape[0]//2 - 1, :, :].cpu().numpy()
    input_1 = input_gt[input_gt.shape[0]//2, :, :].cpu().numpy()
    
    vel_0 = vel_gt[0, :, :].cpu().numpy()
    vel_1 = vel_gt[1, :, :].cpu().numpy()

    vel_0_pred = vel_pred[0, :, :].cpu().numpy()
    vel_1_pred = vel_pred[1, :, :].cpu().numpy()

    if vel_gt.shape[0]==3:
        vel_2 = vel_gt[2, :, :].cpu().numpy()
        vel_2_pred = vel_pred[2, :, :].cpu().numpy()

    vmin = -3 * torch.std(vel_gt)
    vmax = -1 * vmin
    extent = [0, vel_0.shape[1] * 0.016, 0, vel_0.shape[0] * 0.016]
    if arrows:
        H, W = vel_0.shape
        X, Y = np.meshgrid(np.arange(H) * 0.016, np.arange(W) * 0.016, indexing='ij')
        step = 17
        quiver_scale = 10
        quiver_width = 0.01
        quiver_alpha = 0.8


    idx = 0
    if plot_intensity:
        for j, intensity in enumerate([input_0, input_1]):
            im = ax[idx][j].imshow(intensity.T, cmap='magma', origin='lower', extent=extent)
            ax[idx][j].set_title(f'Intensity - channel {j}')
            divider = make_axes_locatable(ax[idx][j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
        idx += 1

    # vx
    im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][0].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0.T[::step, ::step], vel_1.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha,
            #headwidth=6, headlength=8, headaxislength=6.5,
            headwidth=2, headlength=3, headaxislength=2,
            scale_units='xy')
    title_padding = 27
    ax[idx][0].set_title('vx - ground truth', pad=title_padding)
    divider = make_axes_locatable(ax[idx][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    im = ax[idx][1].imshow(vel_0_pred.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][1].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0_pred.T[::step, ::step], vel_1_pred.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha,
            headwidth=2, headlength=3, headaxislength=2,
            scale_units='xy'
        )
    ax[idx][1].set_title('vx - predicted', pad=title_padding)
    divider = make_axes_locatable(ax[idx][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    idx += 1

    # vy

    im = ax[idx][0].imshow(vel_1.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][0].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0.T[::step, ::step], vel_1.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha,
            headwidth=2, headlength=3, headaxislength=2,
            scale_units='xy'
        )
    ax[idx][0].set_title('vy - ground truth', pad=title_padding)
    divider = make_axes_locatable(ax[idx][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    im = ax[idx][1].imshow(vel_1_pred.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if arrows:
        ax[idx][1].quiver(
            Y[::step, ::step], X[::step, ::step],
            vel_0_pred.T[::step, ::step], vel_1_pred.T[::step, ::step],
            color='k', scale=quiver_scale, width=quiver_width, alpha=quiver_alpha,
            headwidth= 2, headlength=3, headaxislength=2,
            scale_units='xy'
        )
    ax[idx][1].set_title('vy - predicted', pad=title_padding)
    divider = make_axes_locatable(ax[idx][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    if vel_gt.shape[0]==3:
        idx += 1
        # vz
        im = ax[idx][0].imshow(vel_2.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
        ax[idx][0].set_title('vz - ground truth', pad=title_padding)
        divider = make_axes_locatable(ax[idx][0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        im = ax[idx][1].imshow(vel_2_pred.T, cmap='bwr', vmin=vmin, vmax=vmax, origin='lower', extent=extent)
        ax[idx][1].set_title('vz - predicted', pad=title_padding)
        divider = make_axes_locatable(ax[idx][1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if arrows:
        plt.savefig(os.path.join(path_save, filename + '_arrows.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path_save, filename + '.png'), bbox_inches='tight')
    plt.close()

def plot_scatter_plot (original_velocity, predicted_velocity, path_save, filename, plotting_dim = None, zoom_out = 0):
    """
    Create scatter plots comparing original and predicted velocity components.
    Args:
        original_velocity: Original velocity tensor (2 or 3-channel)
        predicted_velocity: Predicted velocity tensor (2 or 3-channel)
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
    ncols = 2 if original_velocity_plot.shape[0]==2 else 3
    figsize = (10*ncols, 10)
    fig, ax = plt.subplots(nrows = 1, ncols = ncols, sharex = True, sharey = True, figsize=figsize)

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

    if original_velocity_plot.shape[0]==3:
        im = ax[2].scatter(original_velocity_plot[2].flatten(), predicted_velocity_plot[2].flatten(), alpha = 0.05,linewidths = 0.7)
        min2 = min(original_velocity_plot[2].min(), predicted_velocity_plot[2].min()) - zoom_out
        max2 = max(original_velocity_plot[2].max(), predicted_velocity_plot[2].max()) + zoom_out
        ax[2].plot([min2, max2], [min2, max2], color = 'red') 
        ax[2].set_title('Vz - original vs predicted')
        ax[2].set_xlabel('Original data')
        ax[2].set_ylabel('Predicted data')

    plt.savefig(os.path.join(path_save, filename + '.png'))
    plt.close()

# def plot_test_full_map (deepvel_object, params_model, dataset_path, test_save_path, name, zoomed_in_size = None, plot_intensity = True, n_input_channels = 2, 
#                                                    scatter_dim = None, zoom_out = 0, write_metrics = False, denormalized = False, return_metrics = False, model_type = ModelType.DEEPVEL_I, title = None, tau = None, arrows = False):
#     """
#     Plot predictions and scatter plots for a full map in a vertical layout.
#     Args:
#         deepvel_object: Instance of the DeepVel model
#         params_model: Model parameters
#         dataset_path: Path where the dataset is located
#         test_save_path: Directory to save the test plots
#         name: Base name for the saved plot files
#         zoomed_in_size: Optional tuple specifying the size to zoom in on (height, width)
#         plot_intensity: Boolean indicating whether to plot intensity maps
#         n_input_channels: Number of input channels in the intensity map (2, 4, or 6)
#         scatter_dim: Optional tuple specifying the dimensions to zoom in on for scatter plot (height, width)
#         zoom_out: Value to expand the axes limits for better visualization in scatter plot
#         write_metrics: Boolean indicating whether to write metrics on the plot
#         denormalized: Boolean indicating whether to denormalize velocity data before plotting
#         return_metrics: Boolean indicating whether to return calculated metrics
#         model_type: Enum indicating the model type (DEEPVEL_I, DEEPVEL_B, HYBRID_Bvz, HYBRID_IBvz, STOKES_I, STOKES_V, STOKES_IV)
#         title: Optional title for the plot
#         arrows: Boolean indicating whether to overlay arrows on velocity fields
#         tau: Optional optical depth parameter
#     """

#     plt.rcParams['font.size'] = 40
#     inputs, velocity_to_plot, velocity_pred = load_data_and_get_predictions(deepvel_object, params_model, dataset_path, n_input_channels, model_type, tau, zoomed_in_size)
#     extent = [0, velocity_to_plot.shape[1]*0.016, 0, velocity_to_plot.shape[0]*0.016]

#     intensity_to_plot = inputs["intensity"]
    
#     plot_predictions(
#         intensity_to_plot, velocity_to_plot, velocity_pred,
#         test_save_path, name,
#         plot_intensity=plot_intensity,
#         arrows=arrows
#     )
#     plot_scatter_plot(velocity_to_plot, velocity_pred, test_save_path, 'scatter_'+name)

#     #NOTE: I CAME TO HERE, REFACTOR THE DIV AND VORT
#     divergence_orig = get_divergence(velocity_to_plot[0].cpu().numpy(), velocity_to_plot[1].cpu().numpy())
#     divergence_pred = get_divergence(velocity_pred[0].cpu().numpy(), velocity_pred[1].cpu().numpy())
#     vorticity_orig = get_vorticity(velocity_to_plot[0].cpu().numpy(), velocity_to_plot[1].cpu().numpy())
#     vorticity_pred = get_vorticity(velocity_pred[0].cpu().numpy(), velocity_pred[1].cpu().numpy())

#     mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(velocity_to_plot, velocity_pred).values()
#     mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
#     mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

#     if return_metrics:
#         return {
#             "mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "slope_vx": slope_x, "slope_vy": slope_y, 
#             "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
#             "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v
#        }


def plot_prediction_and_scatter_full_map_vertical (deepvel_object, params_model, dataset_path, test_save_path, name, zoomed_in_size = None, plot_intensity = False, n_input_channels = 2, n_out_channels = 3, 
                                                   scatter_dim = None, zoom_out = 0, write_metrics = False, denormalized = False, mean_vel = None, std_vel = None, return_metrics = False, 
                                                   model_type = ModelType.DEEPVEL_I, title = None, tau = None, sliding_windows = False):
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
        model_type: Enum indicating the model type (DEEPVEL_I, DEEPVEL_B, HYBRID1, HYBRID_Bvz)
        title: Optional title for the plot
        tau: Optional optical depth parameter
    """
    #TODO refactor to work with different shapes of velocities! 
    plt.rcParams['font.size'] = 50

    inputs, velocity_to_plot, velocity_pred = load_data_and_get_predictions(deepvel_object, params_model, dataset_path, 
                                                                            n_input_channels, n_out_channels, model_type, tau, zoomed_in_size, 
                                                                            sliding_windows = sliding_windows)
    intensity_to_plot = inputs["intensity"]

    extent = [0, velocity_to_plot.shape[2]*0.016, 0, velocity_to_plot.shape[1]*0.016]
        
    # print("velocity_to_plot shape:", velocity_to_plot.shape)
    # print("velocity_pred shape:", velocity_pred.shape)
    mse_l = nn.functional.mse_loss(velocity_pred.to(device), velocity_to_plot.to(device))
    rmse_l = torch.sqrt(mse_l)

    matplotlib.use('agg')
    # plt.figure()

    nrows = 4 if plot_intensity else 3
    ncols = velocity_to_plot.shape[0]    

    if velocity_to_plot.shape[0]==3: figsize = (43, 43)
    elif velocity_to_plot.shape[0]==2: figsize=(37, 45)
    else: figsize = (15, 40)

    # fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    # plt.subplots_adjust(hspace=0.4)
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=figsize, squeeze=False)
    plt.subplots_adjust(hspace=0.35, top=0.90, bottom=0.08, right=0.92)

    if plot_intensity == True:
        intensity_0 = intensity_to_plot[n_input_channels/2 -1, :, :].cpu().numpy()
        intensity_1 = intensity_to_plot[n_input_channels/2, :, :].cpu().numpy()

    vel_0 = velocity_to_plot[0, :, :].cpu().numpy()
    vel_0_pred = velocity_pred[0, :, :].cpu().numpy()
    
    print("Velocity to plot shape is: ", velocity_to_plot.shape)
    if velocity_to_plot.shape[0]>1:
        vel_1 = velocity_to_plot[1, :, :].cpu().numpy()
        vel_1_pred = velocity_pred[1, :, :].cpu().numpy()

    if velocity_to_plot.shape[0]==3:
        vel_2 = velocity_to_plot[2, :, :].cpu().numpy()
        vel_2_pred = velocity_pred[2, :, :].cpu().numpy()

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

    if denormalized:
        if std_vel is None:
            std_vel_x = np.std(vel_0)
            std_vel_y = np.std(vel_1) if velocity_to_plot.shape[0]>1 else std_vel_x
            if velocity_to_plot.shape[0]==3:
                std_vel_z = np.std(vel_2)
        else:
            if len(std_vel) > 1:
                std_vel_x = std_vel[0]
                std_vel_y = std_vel[1] if velocity_to_plot.shape[0]>1 else std_vel_x
                vmin = -3*max(std_vel_x, std_vel_y)/1e5
                vmax = -1*vmin
                if velocity_to_plot.shape[0]==3:
                    std_vel_z = std_vel[2]
                    vmin = -3*max(std_vel_x, std_vel_y, std_vel_z)/1e5
                    vmax = -1*vmin
            else:
                vmin = -3*std_vel[0]/1e5
                vmax = -1*vmin

    else:
        vmin = -3*torch.std(velocity_to_plot)
        vmax = -1*vmin

    if denormalized:
        if len(mean_vel) > 1:
            mean_vel_x = mean_vel[0]          
            vel_0 = denormalize_data(vel_0, mean_vel_x, std_vel_x)/1e5
            vel_0_pred = denormalize_data(vel_0_pred, mean_vel_x, std_vel_x)/1e5

            if velocity_to_plot.shape[0]>1:
                mean_vel_y = mean_vel[1]
                vel_1 = denormalize_data(vel_1, mean_vel_y, std_vel_y)/1e5
                vel_1_pred = denormalize_data(vel_1_pred, mean_vel_y, std_vel_y)/1e5

            if velocity_to_plot.shape[0]==3:
                mean_vel_z = mean_vel[2]
                vel_2 = denormalize_data(vel_2, mean_vel_z, std_vel_z)/1e5
                vel_2_pred = denormalize_data(vel_2_pred, mean_vel_z, std_vel_z)/1e5
        else:
            vel_0 = denormalize_data(vel_0, mean_vel[0], std_vel[0])/1e5
            vel_0_pred = denormalize_data(vel_0_pred, mean_vel[0], std_vel[0])/1e5

            if velocity_to_plot.shape[0]>1:
                vel_1 = denormalize_data(vel_1, mean_vel[0], std_vel[0])/1e5
                vel_1_pred = denormalize_data(vel_1_pred, mean_vel[0], std_vel[0])/1e5

            if velocity_to_plot.shape[0]==3:
                vel_2 = denormalize_data(vel_2, mean_vel[0], std_vel[0])/1e5
                vel_2_pred = denormalize_data(vel_2_pred, mean_vel[0], std_vel[0])/1e5

    if velocity_to_plot.shape[0]==1:
        im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][0].set_title('vz - ground truth', pad = 20)
        ax[idx][0].set_xlabel('x [Mm]')
        ax[idx][0].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][0])

    if velocity_to_plot.shape[0]>1:
        im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][0].set_title('vx - ground truth', pad = 20)
        ax[idx][0].set_xlabel('x [Mm]')
        ax[idx][0].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][0])

        ax[idx][1].imshow(vel_1.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][1].set_title('vy - ground truth', pad = 20)
        ax[idx][1].set_xlabel('x [Mm]')
        ax[idx][1].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][1])

    if velocity_to_plot.shape[0]==3:

        ax[idx][2].imshow(vel_2.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][2].set_title('vz - ground truth', pad = 20)
        ax[idx][2].set_xlabel('x [Mm]')
        ax[idx][2].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][2])

    idx +=1

    if velocity_to_plot.shape[0]==1:
        im = ax[idx][0].imshow(vel_0_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][0].set_title('vz - predicted', pad = 20)
        ax[idx][0].set_xlabel('x [Mm]')
        ax[idx][0].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][0])
    
    if velocity_to_plot.shape[0]>1:
        ax[idx][0].imshow(vel_0_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][0].set_title('vx - predicted', pad = 20)
        ax[idx][0].set_xlabel('x [Mm]')
        ax[idx][0].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][0])

        ax[idx][1].imshow(vel_1_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][1].set_title('vy - predicted', pad = 20)
        ax[idx][1].set_xlabel('x [Mm]')
        ax[idx][1].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][1])

    if velocity_to_plot.shape[0]==3:
        ax[idx][2].imshow(vel_2_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
        ax[idx][2].set_title('vz - predicted', pad = 20)
        ax[idx][2].set_xlabel('x [Mm]')
        ax[idx][2].set_ylabel('y [Mm]')
        cbar = add_colorbar(im, ax[idx][2])

    idx +=1

    if velocity_to_plot.shape[0]==1:
        im = ax[idx][0].scatter(vel_0.flatten(), vel_0_pred.flatten(), alpha = 0.05, linewidths = 0.7)
        min0 = min(vel_0.min(), vel_0_pred.min()) - zoom_out
        max0 = max(vel_0.max(), vel_0_pred.max()) + zoom_out
        ax[idx][0].plot([min0, max0], [min0, max0], color = 'red') 
        ax[idx][0].set_title('Vz - original vs predicted', pad = 20)
        ax[idx][0].set_xlabel('Original data')
        ax[idx][0].set_ylabel('Predicted data')
    
    if velocity_to_plot.shape[0]>1:
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

    if velocity_to_plot.shape[0]==3:
        im = ax[idx][2].scatter(vel_2.flatten(), vel_2_pred.flatten(), alpha = 0.05, linewidths = 0.7)
        min2 = min(vel_2.min(), vel_2_pred.min()) - zoom_out
        max2 = max(vel_2.max(), vel_2_pred.max()) + zoom_out
        ax[idx][2].plot([min2, max2], [min2, max2], color = 'red') 
        ax[idx][2].set_title('Vz - original vs predicted', pad = 20)
        ax[idx][2].set_xlabel('Original data')
        ax[idx][2].set_ylabel('Predicted data')
        
    if velocity_to_plot.shape[0]>1:
        divergence_orig = get_divergence(vel_0, vel_1)
        divergence_pred = get_divergence(vel_0_pred, vel_1_pred)
        vorticity_orig = get_vorticity(vel_0, vel_1)
        vorticity_pred = get_vorticity(vel_0_pred, vel_1_pred)

    if velocity_to_plot.shape[0]==1:
        mse_l, rmse_l, pearson2, slope_z = get_all_metrics(velocity_to_plot, velocity_pred).values()
    elif velocity_to_plot.shape[0]==3:
        mse_l, rmse_l, pearson0, pearson1, pearson2, slope_x, slope_y, slope_z = get_all_metrics(velocity_to_plot, velocity_pred).values()
    else:
        mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(velocity_to_plot, velocity_pred).values()
    
    if velocity_to_plot.shape[0]>1:
        mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
        mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

    if write_metrics: 
        if velocity_to_plot.shape[0]==1:
            fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vz: {pearson2:.3f}, Slope Vz: {slope_z:.3f}', 
             ha='center', fontsize=16)
        elif velocity_to_plot.shape[0]==3:
            fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Pearson Vz: {pearson2:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}, Slope Vz: {slope_z:.3f}', 
             ha='center', fontsize=16)
            
        else:   
            fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}', 
            ha='center', fontsize=16)
    
    else: 
        if velocity_to_plot.shape[0]==3:
            print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Pearson Vz: {pearson2:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}, Slope Vz: {slope_z:.3f}')
        elif velocity_to_plot.shape[0]==1:
            print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vz: {pearson2:.3f}, Slope Vz: {slope_z:.3f}')
        else:
            print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}')
        if velocity_to_plot.shape[0]>1:
            print(f'Divergence - MSE Loss: {mse_d:.4f}, Root MSE Loss: {rmse_d:.4f}, Pearson: {pearson_d:.3f}, Slope: {slope_d:.3f}')
            print(f'Vorticity - MSE Loss: {mse_v:.4f}, Root MSE Loss: {rmse_v:.4f}, Pearson: {pearson_v:.3f}, Slope: {slope_v:.3f}')
        
    if title is not None:
        if denormalized: title = title + '\n (denormalized)'
        if sliding_windows: title = title + '\n (sliding windows)'
        plt.suptitle(title, fontsize=70, y=0.98, wrap=True)
    plt.tight_layout()
    if test_save_path is not None:
        if denormalized:
            if sliding_windows:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_denormalized_sliding_windows_' + name + '_.png'))
            else:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_denormalized_' + name + '.png'))
        else:
            if sliding_windows:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_sliding_windows_' + name + '.png'))
            else:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_' + name + '.png'))
    plt.close(fig)

    if return_metrics:
        if velocity_to_plot.shape[0]==1:
            return {
                "mse": mse_l, "rmse": rmse_l, "pearson_vz": pearson2, "slope_vz": slope_z
            }
        if velocity_to_plot.shape[0]==3:
            return {
                "mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "pearson_vz": pearson2, "slope_vx": slope_x, "slope_vy": slope_y, "slope_vz": slope_z,
                "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
                "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v
            }
        return {"mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "slope_vx": slope_x, "slope_vy": slope_y, 
                "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
                "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v}
    

def plot_prediction_and_scatter_full_map_vertical_multiheight (deepvel_object, params_model, dataset_path, test_save_path, name, input_shape, output_shape, zoomed_in_size = None, 
                                                   scatter_dim = None, zoom_out = 0, write_metrics = False, denormalized = False, mean_vels = None, std_vels = None, return_metrics = False, 
                                                   model_type = ModelType.DEEPVEL_I, title = None, taus = None):
    """
    TODO rewrite the description
    # Plot predictions and scatter plots for a full map in a vertical layout.
    # Args:
    #     deepvel_object: Instance of the DeepVel model
    #     params_model: Model parameters
    #     dataset_path: Path where the dataset is located
    #     test_save_path: Directory to save the test plots
    #     name: Base name for the saved plot files
    #     zoomed_in_size: Optional tuple specifying the size to zoom in on (height, width)
    #     plot_intensity: Boolean indicating whether to plot intensity maps
    #     n_input_channels: Number of input channels in the intensity map (2, 4, or 6)
    #     scatter_dim: Optional tuple specifying the dimensions to zoom in on for scatter plot (height, width)
    #     zoom_out: Value to expand the axes limits for better visualization in scatter plot
    #     write_metrics: Boolean indicating whether to write metrics on the plot
    #     denormalized: Boolean indicating whether to denormalize velocity data before plotting
    #     return_metrics: Boolean indicating whether to return calculated metrics
    #     model_type: Enum indicating the model type (DEEPVEL_I, DEEPVEL_B, HYBRID1, HYBRID_Bvz)
    #     title: Optional title for the plot
    #     tau: Optional optical depth parameter
    """
    plt.rcParams['font.size'] = 50
    n_input_channels = input_shape[0]
    n_out_channels = output_shape[1]

    velocity_gt = []
    for t in taus:
        velocity_full_map = np.load(os.path.join(dataset_path, f"labels/tau_{t}/velocities_tau_{t}.npy"))[0, :, :, :]
        velocity_gt.append(torch.from_numpy(velocity_full_map).float().to(device))
    velocity_gt = torch.stack(velocity_gt, dim =0)

    if model_type == ModelType.STOKES_I:
        input = np.load(os.path.join(dataset_path, f"inputs/stokes_I/stokes_I_{n_input_channels}.npy"))
        velocity_prediction = deepvel_object.predict(input, params_model)
    elif model_type == ModelType.STOKES_IV:
        input1 = np.load(os.path.join(dataset_path, f"inputs/stokes_I/stokes_I_{n_input_channels}.npy"))
        input2 = np.load(os.path.join(dataset_path, f"inputs/stokes_V/stokes_V_{n_input_channels}.npy"))
        velocity_prediction = deepvel_object.predict(input1, input2, params_model)

    for i in range(velocity_gt.shape[0]):
        velocity_to_plot = velocity_gt[i, :, :]
        velocity_pred = velocity_prediction[i, :, :]
        mean_vel = mean_vels[taus[i]] if mean_vels is not None else None
        std_vel = std_vels[taus[i]] if std_vels is not None else None
        extent = [0, velocity_to_plot.shape[2]*0.016, 0, velocity_to_plot.shape[1]*0.016]
            

        mse_l = nn.functional.mse_loss(velocity_pred.to(device), velocity_to_plot.to(device))
        rmse_l = torch.sqrt(mse_l)

        matplotlib.use('agg')

        nrows = 3
        ncols = velocity_to_plot.shape[0]    

        if velocity_to_plot.shape[0]==3: figsize = (43, 43)
        elif velocity_to_plot.shape[0]==2: figsize=(37, 45)
        else: figsize = (15, 40)


        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=figsize, squeeze=False)
        plt.subplots_adjust(hspace=0.35, top=0.90, bottom=0.08, right=0.92)

        vel_0 = velocity_to_plot[0, :, :].cpu().numpy()
        vel_0_pred = velocity_pred[0, :, :].cpu().numpy()
        
        print("Velocity to plot shape is: ", velocity_to_plot.shape)
        if velocity_to_plot.shape[0]>1:
            vel_1 = velocity_to_plot[1, :, :].cpu().numpy()
            vel_1_pred = velocity_pred[1, :, :].cpu().numpy()

        if velocity_to_plot.shape[0]==3:
            vel_2 = velocity_to_plot[2, :, :].cpu().numpy()
            vel_2_pred = velocity_pred[2, :, :].cpu().numpy()

        idx = 0

        #TODO intensity denormalization option


        if denormalized:
            if std_vel is None:
                std_vel_x = np.std(vel_0)
                std_vel_y = np.std(vel_1) if velocity_to_plot.shape[0]>1 else std_vel_x
                if velocity_to_plot.shape[0]==3:
                    std_vel_z = np.std(vel_2)
           
            else:
                    vmin = -3*std_vel/1e5
                    vmax = -1*vmin

        else:
            vmin = -3*torch.std(velocity_to_plot)
            vmax = -1*vmin

        if denormalized:
                vel_0 = denormalize_data(vel_0, mean_vel, std_vel)/1e5
                vel_0_pred = denormalize_data(vel_0_pred, mean_vel, std_vel)/1e5

                if velocity_to_plot.shape[0]>1:
                    vel_1 = denormalize_data(vel_1, mean_vel, std_vel)/1e5
                    vel_1_pred = denormalize_data(vel_1_pred, mean_vel, std_vel)/1e5

                if velocity_to_plot.shape[0]==3:
                    vel_2 = denormalize_data(vel_2, mean_vel, std_vel)/1e5
                    vel_2_pred = denormalize_data(vel_2_pred, mean_vel, std_vel)/1e5

        if velocity_to_plot.shape[0]==1:
            im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][0].set_title('vz - ground truth', pad = 20)
            ax[idx][0].set_xlabel('x [Mm]')
            ax[idx][0].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][0])

        if velocity_to_plot.shape[0]>1:
            im = ax[idx][0].imshow(vel_0.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][0].set_title('vx - ground truth', pad = 20)
            ax[idx][0].set_xlabel('x [Mm]')
            ax[idx][0].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][0])

            ax[idx][1].imshow(vel_1.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][1].set_title('vy - ground truth', pad = 20)
            ax[idx][1].set_xlabel('x [Mm]')
            ax[idx][1].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][1])

        if velocity_to_plot.shape[0]==3:

            ax[idx][2].imshow(vel_2.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][2].set_title('vz - ground truth', pad = 20)
            ax[idx][2].set_xlabel('x [Mm]')
            ax[idx][2].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][2])

        idx +=1

        if velocity_to_plot.shape[0]==1:
            im = ax[idx][0].imshow(vel_0_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][0].set_title('vz - predicted', pad = 20)
            ax[idx][0].set_xlabel('x [Mm]')
            ax[idx][0].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][0])
        
        if velocity_to_plot.shape[0]>1:
            ax[idx][0].imshow(vel_0_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][0].set_title('vx - predicted', pad = 20)
            ax[idx][0].set_xlabel('x [Mm]')
            ax[idx][0].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][0])

            ax[idx][1].imshow(vel_1_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][1].set_title('vy - predicted', pad = 20)
            ax[idx][1].set_xlabel('x [Mm]')
            ax[idx][1].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][1])

        if velocity_to_plot.shape[0]==3:
            ax[idx][2].imshow(vel_2_pred.T, cmap='bwr', vmin = vmin, vmax = vmax, origin = 'lower', extent = extent, aspect='auto')
            ax[idx][2].set_title('vz - predicted', pad = 20)
            ax[idx][2].set_xlabel('x [Mm]')
            ax[idx][2].set_ylabel('y [Mm]')
            cbar = add_colorbar(im, ax[idx][2])

        idx +=1

        if velocity_to_plot.shape[0]==1:
            im = ax[idx][0].scatter(vel_0.flatten(), vel_0_pred.flatten(), alpha = 0.05, linewidths = 0.7)
            min0 = min(vel_0.min(), vel_0_pred.min()) - zoom_out
            max0 = max(vel_0.max(), vel_0_pred.max()) + zoom_out
            ax[idx][0].plot([min0, max0], [min0, max0], color = 'red') 
            ax[idx][0].set_title('Vz - original vs predicted', pad = 20)
            ax[idx][0].set_xlabel('Original data')
            ax[idx][0].set_ylabel('Predicted data')
        
        if velocity_to_plot.shape[0]>1:
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

        if velocity_to_plot.shape[0]==3:
            im = ax[idx][2].scatter(vel_2.flatten(), vel_2_pred.flatten(), alpha = 0.05, linewidths = 0.7)
            min2 = min(vel_2.min(), vel_2_pred.min()) - zoom_out
            max2 = max(vel_2.max(), vel_2_pred.max()) + zoom_out
            ax[idx][2].plot([min2, max2], [min2, max2], color = 'red') 
            ax[idx][2].set_title('Vz - original vs predicted', pad = 20)
            ax[idx][2].set_xlabel('Original data')
            ax[idx][2].set_ylabel('Predicted data')
            
        if velocity_to_plot.shape[0]>1:
            divergence_orig = get_divergence(vel_0, vel_1)
            divergence_pred = get_divergence(vel_0_pred, vel_1_pred)
            vorticity_orig = get_vorticity(vel_0, vel_1)
            vorticity_pred = get_vorticity(vel_0_pred, vel_1_pred)

        if velocity_to_plot.shape[0]==1:
            mse_l, rmse_l, pearson2, slope_z = get_all_metrics(velocity_to_plot, velocity_pred).values()
        elif velocity_to_plot.shape[0]==3:
            mse_l, rmse_l, pearson0, pearson1, pearson2, slope_x, slope_y, slope_z = get_all_metrics(velocity_to_plot, velocity_pred).values()
        else:
            mse_l, rmse_l, pearson0, pearson1, slope_x, slope_y = get_all_metrics(velocity_to_plot, velocity_pred).values()
        
        if velocity_to_plot.shape[0]>1:
            mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
            mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

        if write_metrics: 
            if velocity_to_plot.shape[0]==1:
                fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vz: {pearson2:.3f}, Slope Vz: {slope_z:.3f}', 
                ha='center', fontsize=16)
            elif velocity_to_plot.shape[0]==3:
                fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Pearson Vz: {pearson2:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}, Slope Vz: {slope_z:.3f}', 
                ha='center', fontsize=16)
                
            else:   
                fig.text(0.5, 0.05, f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}', 
                ha='center', fontsize=16)
        
        else: 
            if velocity_to_plot.shape[0]==3:
                print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Pearson Vz: {pearson2:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}, Slope Vz: {slope_z:.3f}')
            elif velocity_to_plot.shape[0]==1:
                print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vz: {pearson2:.3f}, Slope Vz: {slope_z:.3f}')
            else:
                print(f'MSE Loss: {mse_l:.4f}, Root MSE Loss: {rmse_l:.4f}, Pearson Vx: {pearson0:.3f}, Pearson Vy: {pearson1:.3f}, Slope Vx: {slope_x:.3f}, Slope Vy: {slope_y:.3f}')
            if velocity_to_plot.shape[0]>1:
                print(f'Divergence - MSE Loss: {mse_d:.4f}, Root MSE Loss: {rmse_d:.4f}, Pearson: {pearson_d:.3f}, Slope: {slope_d:.3f}')
                print(f'Vorticity - MSE Loss: {mse_v:.4f}, Root MSE Loss: {rmse_v:.4f}, Pearson: {pearson_v:.3f}, Slope: {slope_v:.3f}')
            
        if title is not None:
            if denormalized:
                plt.suptitle(title + " - tau: " + str(taus[i]) + '\n (denormalized)', fontsize=70, y=0.98, wrap=True)
            else:
                plt.suptitle(title + " - tau: " + str(taus[i]), fontsize=70, y=0.98, wrap=True)
        plt.tight_layout()
        if test_save_path is not None:
            if denormalized:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_denormalized_' + name + '_tau_' + str(taus[i]) + '.png'))
            else:
                fig.savefig(os.path.join(test_save_path, 'full_analysis_' + name + '_tau_' + str(taus[i]) + '.png'))
        plt.close(fig)

        if return_metrics:
            if velocity_to_plot.shape[0]==1:
                return {
                    "mse": mse_l, "rmse": rmse_l, "pearson_vz": pearson2, "slope_vz": slope_z
                }
            if velocity_to_plot.shape[0]==3:
                return {
                    "mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "pearson_vz": pearson2, "slope_vx": slope_x, "slope_vy": slope_y, "slope_vz": slope_z,
                    "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
                    "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v
                }
            return {"mse": mse_l, "rmse": rmse_l, "pearson_vx": pearson0, "pearson_vy": pearson1, "slope_vx": slope_x, "slope_vy": slope_y, 
                    "mse_divergence": mse_d, "rmse_divergence": rmse_d, "pearson_divergence": pearson_d, "slope_divergence": slope_d,
                    "mse_vorticity": mse_v, "rmse_vorticity": rmse_v, "pearson_vorticity": pearson_v, "slope_vorticity": slope_v}