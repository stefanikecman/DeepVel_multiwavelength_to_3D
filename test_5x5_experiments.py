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
import csv
#from prepare_data import normalize_layerwise
from metrics_and_plotting import plot_predictions, plot_test_full_map, plot_scatter_plot, calculate_correlation, plot_prediction_and_scatter_full_map, plot_prediction_and_scatter_full_map_vertical
from deepvel_torch import DeepVel_run
import pandas as pd
import seaborn as sb
from scipy.stats import norm

def plot_metric_heatmaps(csv_path, save_dir):

    df = pd.read_csv(csv_path)
    metrics = ["mse", "rmse", "pearson_vx", "pearson_vy", "slope_vx", "slope_vy"]

    for metric in metrics:
        pivot_table = df.pivot(index="timesteps", columns="patch_size", values=metric)

        plt.figure(figsize=(10, 8))
        sb.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
        plt.title(f"{metric.upper()} by Timesteps and Patch Size", fontsize=14)
        plt.xlabel("Patch Size")
        plt.ylabel("Timesteps")
        
        save_path = os.path.join(save_dir, f"heatmap_{metric}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

def get_divergence(vx, vy, dx, dy):

    if isinstance(vx, torch.Tensor):
        vx = vx.detach().cpu().numpy()

    if isinstance(vy, torch.Tensor):
        vy = vy.detach().cpu().numpy()

    dvx_dx = np.gradient(vx, dx, axis=1)
    dvy_dy = np.gradient(vy, dy, axis=0)

    return dvx_dx + dvy_dy

def get_vorticity(vx, vy, dx, dy):

    if isinstance(vx, torch.Tensor):
        vx = vx.detach().cpu().numpy()

    if isinstance(vy, torch.Tensor):
        vy = vy.detach().cpu().numpy()

    dvy_dx = np.gradient(vy, dx, axis=1)
    dvx_dy = np.gradient(vx, dy, axis=0)

    return dvy_dx - dvx_dy


def compare_div_hel (params_model, intensities, original_velocities):
    # NOTE I get dx and dy by knowing the resolution and size of the map

    _, H, W = original_velocities.shape
    Lx = W * 0.016 # in Mm
    Ly = H * 0.016 # in Mm
    dx = dy = 0.016 * 1e6  # in meters

    div_original = get_divergence(original_velocities[0], original_velocities[1], dx, dy)
    vor_original = get_vorticity(original_velocities[0], original_velocities[1], dx, dy)

    main_root = "/dat/xenoss/"
    model = DeepVel_run(root = main_root, in_channels=4, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/normalized/timestep_4/cropped/data_128x128', network_path = main_root + f'models_5x5_norm/timestep_4/128x128/')

    velocity_pred = model.predict(intensities, params_model)
    div_pred = get_divergence(velocity_pred[0], velocity_pred[1], dx, dy)
    vor_pred = get_vorticity(velocity_pred[0], velocity_pred[1], dx, dy)

    # Compare divergence and vorticity
    div_diff = div_pred - div_original
    vor_diff = vor_pred - vor_original

    # print("Original Divergence:", div_original)
    # print("Original Vorticity:", vor_original)
    # print("Predicted Divergence:", div_pred)
    # print("Predicted Vorticity:", vor_pred)
    # print("Divergence Difference:", div_diff)
    # print("Vorticity Difference:", vor_diff)

    with open("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/numerical_comparison.txt", "w") as f:
        f.write(f"Original Divergence: {div_original}\n")
        f.write(f"Original Vorticity: {vor_original}\n")
        f.write(f"Predicted Divergence: {div_pred}\n")
        f.write(f"Predicted Vorticity: {vor_pred}\n")
        f.write(f"Divergence Difference: {div_diff}\n")
        f.write(f"Vorticity Difference: {vor_diff}\n")

    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/div_original.npy", div_original)
    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/vor_original.npy", vor_original)
    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/div_pred.npy", div_pred)
    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/vor_pred.npy", vor_pred)
    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/div_diff.npy", div_diff)
    np.save("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/vor_diff.npy", vor_diff)


def plot_histograms_div_vor(div_original, vor_original, div_pred, vor_pred):
    main_folder = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/"  # Folder to save both plots and txt
    output_folder = main_folder + "histograms/"
    stats_file = os.path.join(output_folder, "hist_stats.txt")

    bins = 100
    with open(stats_file, "w") as stats_out:
        stats_out.write("File\t\t\tMu (mean)\tSigma (std)\n")
        stats_out.write("="*50 + "\n")

        div_original = div_original.flatten()
        vor_original = vor_original.flatten()
        div_pred = div_pred.flatten()
        vor_pred = vor_pred.flatten()

        for data, label in zip([div_original, vor_original, div_pred, vor_pred], ["div_original", "vor_original", "div_pred", "vor_pred"]):
            mu = np.mean(data)
            std = np.std(data)

            plt.figure(figsize=(10, 8))
            counts, bins_edges, _ = plt.hist(data, bins=bins, density=True, alpha=0.6, color='cornflowerblue', label=label)
            x = np.linspace(*plt.xlim(), 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\nμ={mu:.3e}, σ={std:.3e}')
            plt.title(f"Histogram with Gaussian Fit: {label}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()

            # Save plot
            save_path = os.path.join(output_folder, f"{label}_histogram.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Save stats to txt
            stats_out.write(f"{label:20s} {mu:.6e}   {std:.6e}\n")
            print(f"Saved histogram and stats for {label}")

if (__name__ == '__main__'):


    main_root = "/dat/xenoss/"

    #plot_metric_heatmaps("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/best_checkpoints.csv", "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/heatmaps")
    # df = pd.read_csv("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/best_checkpoints.csv")
    # df.to_excel("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/best_checkpoints.xlsx", sheet_name="Metrics", index=False)

    # best_checkpoints = [
    # {"timesteps": 2, "patch_size": 32, "checkpoint": "DeepVel_torch_epoch_74_0.20240.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 2, "patch_size": 48, "checkpoint": "DeepVel_torch_epoch_77_0.17641.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 2, "patch_size": 64, "checkpoint": "DeepVel_torch_epoch_74_0.16484.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 2, "patch_size": 96, "checkpoint": "DeepVel_torch_epoch_78_0.14897.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 2, "patch_size": 128, "checkpoint": "DeepVel_torch_epoch_79_0.14283.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},

    # {"timesteps": 4, "patch_size": 32, "checkpoint": "DeepVel_torch_epoch_76_0.20351.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 4, "patch_size": 48, "checkpoint": "DeepVel_torch_epoch_79_0.17468.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 4, "patch_size": 64, "checkpoint": "DeepVel_torch_epoch_77_0.16255.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 4, "patch_size": 96, "checkpoint": "DeepVel_torch_epoch_79_0.15021.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 4, "patch_size": 128, "checkpoint": "DeepVel_torch_epoch_79_0.14127.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},

    # {"timesteps": 6, "patch_size": 32, "checkpoint": "DeepVel_torch_epoch_78_0.20556.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 6, "patch_size": 48, "checkpoint": "DeepVel_torch_epoch_78_0.17807.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 6, "patch_size": 64, "checkpoint": "DeepVel_torch_epoch_79_0.16362.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 6, "patch_size": 96, "checkpoint": "DeepVel_torch_epoch_78_0.14970.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 6, "patch_size": 128, "checkpoint": "DeepVel_torch_epoch_78_0.14175.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},

    # {"timesteps": 8, "patch_size": 32, "checkpoint": "DeepVel_torch_epoch_78_0.20410.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 8, "patch_size": 48, "checkpoint": "DeepVel_torch_epoch_79_0.18011.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 8, "patch_size": 64, "checkpoint": "DeepVel_torch_epoch_78_0.16477.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 8, "patch_size": 96, "checkpoint": "DeepVel_torch_epoch_79_0.15058.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 8, "patch_size": 128, "checkpoint": "DeepVel_torch_epoch_78_0.14107.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},

    # {"timesteps": 10, "patch_size": 32, "checkpoint": "DeepVel_torch_epoch_77_0.20689.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 10, "patch_size": 48, "checkpoint": "DeepVel_torch_epoch_77_0.17687.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 10, "patch_size": 64, "checkpoint": "DeepVel_torch_epoch_78_0.16510.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 10, "patch_size": 96, "checkpoint": "DeepVel_torch_epoch_78_0.14924.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # {"timesteps": 10, "patch_size": 128, "checkpoint": "DeepVel_torch_epoch_79_0.14180.pt", "mse": None, "rmse": None, "pearson_vx": None, "pearson_vy": None, "slope_vx": None, "slope_vy": None},
    # ]

    # unique_timesteps = sorted({el["timesteps"] for el in best_checkpoints})
    # unique_patches = sorted({el["patch_size"] for el in best_checkpoints})

    # for timestep in unique_timesteps:
    #     for patch in unique_patches:
    #         deepvel_net = DeepVel_run(root = main_root, in_channels=timestep, batch = 64, dataset_path = main_root + f'datasets_5x5_experiments/normalized/timestep_{timestep}/cropped/data_{patch}x{patch}', network_path = main_root + f'models_5x5_norm/timestep_{timestep}/{patch}x{patch}/')
    #         params_model = main_root + f'models_5x5_norm/timestep_{timestep}/{patch}x{patch}/' + [el for el in best_checkpoints if el["timesteps"]==timestep and el["patch_size"]==patch][0]["checkpoint"]
    #         print(f"Evaluating model with {timestep} timesteps and patch size {patch}")
    #         plot_prediction_and_scatter_full_map_vertical(timestep, deepvel_net, params_model, dataset_path = main_root + f'datasets_5x5_experiments/main_dataset_test/normalized/', test_save_path='/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/', name = f"experiment_5x5_not_zoomed_in_full_map_with_I_t{timestep}_p{patch}", n_input_channels=timestep, write_metrics=False, return_metrics=False)
    #         metrics = plot_prediction_and_scatter_full_map_vertical(timestep, deepvel_net, params_model, dataset_path = main_root + f'datasets_5x5_experiments/main_dataset_test/normalized/', test_save_path='/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/', name = f"experiment_5x5_not_zoomed_in_full_map_t{timestep}_p{patch}", n_input_channels=timestep, plot_intensity = False, write_metrics=False, return_metrics=True)
    #         for entry in best_checkpoints:
    #             if entry["timesteps"] == timestep and entry["patch_size"] == patch:
    #                 entry.update(metrics)
    #                 break
    #         print(metrics)

    # with open("best_checkpoints.csv", "w", newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=best_checkpoints[0].keys())
    #     writer.writeheader()
    #     writer.writerows(best_checkpoints)

    ## NOTE Divergence and Vorticity comparison

    # params_model = main_root + f'models_5x5_norm/timestep_4/128x128/' + "DeepVel_torch_epoch_79_0.14127.pt"
    # intensities = np.load(main_root + f'datasets_5x5_experiments/main_dataset_test/normalized/' + f"intensities_4.npy")
    # velocities = np.load(main_root + f'datasets_5x5_experiments/main_dataset_test/normalized/' + f"velocities_4.npy")

    # compare_div_hel(params_model, intensities, velocities)

    # div_original = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/div_original.npy")
    # div_pred = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/div_pred.npy")
    # vor_original = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/vor_original.npy")
    # vor_pred = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/experiment_5x5_test/div_and_vor_comparison/vor_pred.npy")

    # plot_histograms_div_vor(div_original, vor_original, div_pred, vor_pred)

    print("Alpha div ", 1/1.384507e-05)
    print("Alpha vor ", 1/1.434438e-05)