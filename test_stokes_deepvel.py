from curses import version
import numpy as np
import torch
import os
import csv
from collections import OrderedDict
from deepvel_torch import DeepVel_run
import pandas as pd
from analysis_fn import get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType, plot_prediction_and_scatter_full_map_vertical, plot_train_val_losses, infer_with_sliding_windows

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/thesis/"


metrics_by_tau = OrderedDict()

################## NOTE PLOTTING LOSS CURVES FOR STOKES MODELS #####################
# tau = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4","1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
# model_types = [ModelType.STOKES_I, ModelType.STOKES_I, ModelType.STOKES_I, ModelType.STOKES_I,ModelType.STOKES_I, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V]
# version = "v1"
# ckpts_path = os.path.join(main_root, f'thesis/models/{version}/')
# plot_train_val_losses(tau_levels=tau, ckpts_path=ckpts_path, model_type=model_types, title = "Stokes Models Train and Validation Losses", name='stokes_deepvel_models_combined', save_path=ckpts_path)

#######################################################################################


########## NOTE TESTING STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################
# checkpoints_V = ["/home/xenoss/dat/thesis/models/v8/tau_1e-1/stokes_v_vz_model/checkpoints/DeepVel_torch_epoch_170_0.09966.pt",
#                  "/home/xenoss/dat/thesis/models/v8/tau_1e-2/stokes_v_vz_model/checkpoints/DeepVel_torch_epoch_162_0.10630.pt",
#                  "/home/xenoss/dat/thesis/models/v8/tau_1e-3/stokes_v_vz_model/checkpoints/DeepVel_torch_epoch_181_0.17305.pt",
#                  "/home/xenoss/dat/thesis/models/v8/tau_1e-4/stokes_v_vz_model/checkpoints/DeepVel_torch_epoch_164_0.26598.pt"]
# checkpoints_I = ["/home/xenoss/dat/thesis/models/v1/tau_1.0/project_model/checkpoints/DeepVel_torch_epoch_97_0.26407.pt"]
checkpoints_I = ["/home/xenoss/dat/thesis/models/v1/tau_1.0/project_intensities_model_3D/checkpoints/DeepVel_torch_epoch_96_0.25060.pt"]
checkpoints = checkpoints_I
taus = ["1.0"]
model_names = ["DeepVel_project_model"]
version = "v1"
dataset_path = f'/home/xenoss/dat/thesis/data/{version}/dataset/train/'
test_path = f'/home/xenoss/dat/thesis/data/{version}/dataset/test/last_2_layers/'
model_paths = ["/home/xenoss/dat/thesis/models/"]
model_types = [ModelType.DEEPVEL_I]
input_shape = (2, 43, 64, 64)
output_shape = (3, 64, 64)
stokes_profile = "I"
# csvs = [f"stokes_{stokes_profile.lower()}_vz_last2_tau_{taus[i]}" for i in range(len(taus))]
csvs = [f"project_deepvel_3D_last2_tau_{taus[i]}" for i in range(len(taus))]
common_csv = []
for i, ckpt in enumerate(checkpoints):

    # deepvel_net = DeepVel_run(root = main_root, tau = taus[i], in_shape = input_shape, out_shape = output_shape, batch = 1, dataset_path = dataset_path, 
    #                           network_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/stokes_{stokes_profile.lower()}_model/checkpoints/", 
    #                           stokes_profile=stokes_profile, test=True)

    deepvel_net = DeepVel_run(root = main_root, tau = taus[i], in_shape = input_shape, out_shape = output_shape, batch = 1, dataset_path = dataset_path, 
                              network_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/project_intensities_model_3D/checkpoints/", 
                              stokes_profile=stokes_profile, test=True)
        
    params_model = ckpt

    if model_types[0] == ModelType.STOKES_I:
        input_path = test_path + f"inputs/stokes_I/"

    elif model_types[0] == ModelType.STOKES_V:
        input_path = test_path + f"inputs/stokes_V/"
    
    elif model_types[0] == ModelType.DEEPVEL_I:
        input_path = test_path + f"inputs/intensities/"

    input_test_data = os.listdir(input_path)
    input_test_data.sort()
    vel_path = test_path + f"labels/tau_{taus[i]}/"
    vel_test_data = os.listdir(vel_path)
    vel_test_data.sort()

    metrics_rows = []
    print(f"Evaluating model for tau = {taus[i]} ...")
    for j in range(len(input_test_data)):
        input = torch.from_numpy(np.load(os.path.join(input_path, input_test_data[j])).astype(np.float32))#.unsqueeze(0) # add batch dimension
        ground_truth = torch.from_numpy(np.load(os.path.join(vel_path, vel_test_data[j])).astype(np.float32))[0]#[:2, :, :]#.unsqueeze(0) #only take vz for gt since model is only predicting vz
        print("GT shape:", ground_truth.shape)
        print("Input shape:", input.shape)
        prediction = deepvel_net.predict(input,params_model)  
        print("Prediction shape:", prediction.shape)  
        # prediction = infer_with_sliding_windows(deepvel_net, params_model, input=torch.from_numpy(np.array([input])), window = 256)

        divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
        divergence_pred = get_divergence(prediction[0], prediction[1])
        vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
        vorticity_pred = get_vorticity(prediction[0], prediction[1])

        # mse_l, rmse_l, pearson2, slope_z = get_all_metrics(ground_truth, prediction).values()
        # print("GT shape:", ground_truth.shape)
        # mse_l, rmse_l, pearson0, pearson1,slope_x, slope_y = get_all_metrics(ground_truth, prediction).values()
        mse_l, rmse_l, pearson0, pearson1, pearson2, slope_x, slope_y, slope_z = get_all_metrics(ground_truth, prediction).values()
        mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
        mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

        row = {
            'tau': taus[i],
            'sample_index': j,
            'mse_vel': mse_l,
            'rmse_vel': rmse_l,
            'pearson_vel_x': pearson0,
            'pearson_vel_y': pearson1,
            'pearson_vel_z': pearson2,
            'slope_vel_x': slope_x,
            'slope_vel_y': slope_y,
            'slope_vel_z': slope_z,
            'mse_div': mse_d,
            'rmse_div': rmse_d,
            'pearson_div': pearson_d,
            'slope_div': slope_d,
            'mse_vort': mse_v,
            'rmse_vort': rmse_v,
            'pearson_vort': pearson_v,
            'slope_vort': slope_v
        }
        metrics_rows.append(row)

    csv_dir = os.path.join(main_root, f'models/{version}/test_metrics/')

    df_tau = pd.DataFrame(metrics_rows)
    avg_series = df_tau.select_dtypes(include=[np.number]).mean()
    avg_row = avg_series.to_dict()
    avg_row['tau'] = taus[i]
    avg_row['sample_index'] = 'AVERAGE'
    common_csv.append(avg_row)
    df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path_tau = os.path.join(csv_dir, f'{csvs[i]}_metrics.csv')
    df_tau.to_csv(csv_path_tau, index=False)

csv_path_tau_all = os.path.join(csv_dir, f'project_deepvel_3D_all_taus_metrics.csv')
df_tau_all = pd.DataFrame(common_csv)
df_tau_all.to_csv(csv_path_tau_all, index=False)
#######################################################################################

########## NOTE PLOTTING TEST OF STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################
#TESTING DEEPVEL_NET3, v5:

test_path = f"/home/xenoss/dat/thesis/data/{version}/dataset/test/last_2_layers/"
mean = {
    "1.0": -1595.796630859375,
    "1e-1": -2357.634521484375,
    "1e-2": -1106.9403076171875,
    "1e-3": 655.9984741210938,
    "1e-4": -227.47628784179688
}

std = {
    "1.0": 231733.59375,
    "1e-1": 195649.671875,
    "1e-2": 146830.328125,  
    "1e-3": 116736.734375,
    "1e-4": 132994.109375
}

# NOTE FOR V5, V6_1, V7, V8
# mean = {
#     # "1.0": -1595.796630859375, #note for v8_2
#     "1e-1": -2307.059326171875,
#     "1e-2": -1066.4248046875,
#     "1e-3": 705.2369995117188,
#     "1e-4": -168.59375
# }

# std = {
#     # "1.0": 231733.59375, #note for v8_2
#     "1e-1": 196654.65625,
#     "1e-2": 147546.765625,
#     "1e-3": 118183.453125,
#     "1e-4": 135827.203125
# }

# # NOTE FOR V6_2
# mean = {
#     "1e-1": -2357.634521484375,
# }

# std = {
#     "1e-1": 195649.671875,
# }

# for i, tau in enumerate(taus):

#     model_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/checkpoints/"
#     test_save_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/test/"
#     deepvel_net = DeepVel_run(root = main_root, in_shape=input_shape, out_shape=output_shape, batch = 1, dataset_path = dataset_path, network_path = model_path, 
#                               tau=taus[i], stokes_profile=stokes_profile, test=True)
#     params_model = checkpoints[i]
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     plot_prediction_and_scatter_full_map_vertical(deepvel_object = deepvel_net, params_model = params_model, 
#                 dataset_path = test_path, test_save_path = test_save_path, 
#                 name = f"{model_names[0]}_{taus[i]}", zoomed_in_size = None, 
#                 plot_intensity = False, n_input_channels = input_shape[0], n_out_channels = output_shape[0], scatter_dim = None, zoom_out = 0, 
#                 write_metrics = False,denormalized=True, mean_vel=[mean[taus[i]]], std_vel=[std[taus[i]]], return_metrics = False, model_type = model_types[0], 
#                 title = model_names[0]+", tau = " + taus[i], tau = taus[i], sliding_windows=False)
    
# for i, tau in enumerate(taus):

#     model_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/checkpoints/"
#     test_save_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/test/"
#     deepvel_net = DeepVel_run(root = main_root, in_shape=input_shape, out_shape=output_shape, batch = 1, dataset_path = dataset_path, network_path = model_path, 
#                               tau=taus[i], stokes_profile=stokes_profile, test=True)
#     params_model = checkpoints[i]
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     plot_prediction_and_scatter_full_map_vertical(deepvel_object = deepvel_net, params_model = params_model, 
#                 dataset_path = test_path, test_save_path = test_save_path, 
#                 name = f"{model_names[0]}_{taus[i]}", zoomed_in_size = None, 
#                 plot_intensity = False, n_input_channels = input_shape[0], n_out_channels = output_shape[0], scatter_dim = None, zoom_out = 0, 
#                 write_metrics = False,denormalized=False, mean_vel=[mean[taus[i]]], std_vel=[std[taus[i]]], return_metrics = False, model_type = model_types[0], 
#                 title = model_names[0]+", tau = " + taus[i], tau = taus[i], sliding_windows=False)
    
#######################################################################################