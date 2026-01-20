from curses import version
import numpy as np
import torch
import os
import csv
from collections import OrderedDict
from deepvel_torch import DeepVel_run
import pandas as pd
from analysis_fn import get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType, plot_prediction_and_scatter_full_map_vertical, plot_train_val_losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/thesis/"
dataset_path = '/home/xenoss/dat/thesis/data/v1/dataset/train/'

metrics_by_tau = OrderedDict()

################## NOTE PLOTTING LOSS CURVES FOR STOKES MODELS #####################
# tau = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4","1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
# model_types = [ModelType.STOKES_I, ModelType.STOKES_I, ModelType.STOKES_I, ModelType.STOKES_I,ModelType.STOKES_I, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V]
# version = "v1"
# ckpts_path = os.path.join(main_root, f'thesis/models/{version}/')
# plot_train_val_losses(tau_levels=tau, ckpts_path=ckpts_path, model_type=model_types, title = "Stokes Models Train and Validation Losses", name='stokes_deepvel_models_combined', save_path=ckpts_path)

#######################################################################################


########## NOTE TESTING STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################
test_path = '/home/xenoss/dat/thesis/data/v1/dataset/test/last_2_layers/'

# checkpoints_I = [
#     "/home/xenoss/dat/thesis/models/v1/tau_1.0/stokes_i_model/checkpoints/DeepVel_torch_epoch_188_0.08415.pt", #tau=1.0
#     "/home/xenoss/dat/thesis/models/v1/tau_1e-1/stokes_i_model/checkpoints/DeepVel_torch_epoch_160_0.04374.pt",
#     "/home/xenoss/dat/thesis/models/v1/tau_1e-2/stokes_i_model/checkpoints/DeepVel_torch_epoch_189_0.06733.pt",
#     "/home/xenoss/dat/thesis/models/v1/tau_1e-3/stokes_i_model/checkpoints/DeepVel_torch_epoch_196_0.15958.pt",
#     "/home/xenoss/dat/thesis/models/v1/tau_1e-4/stokes_i_model/checkpoints/DeepVel_torch_epoch_194_0.22924.pt"
#     ]

checkpoints_V = [
    "/home/xenoss/dat/thesis/models/v1/tau_1.0/stokes_v_model/checkpoints/DeepVel_torch_epoch_160_0.18493.pt",
    "/home/xenoss/dat/thesis/models/v1/tau_1e-1/stokes_v_model/checkpoints/DeepVel_torch_epoch_173_0.09268.pt",
    "/home/xenoss/dat/thesis/models/v1/tau_1e-2/stokes_v_model/checkpoints/DeepVel_torch_epoch_132_0.11960.pt",
    "/home/xenoss/dat/thesis/models/v1/tau_1e-3/stokes_v_model/checkpoints/DeepVel_torch_epoch_143_0.24695.pt",
    "/home/xenoss/dat/thesis/models/v1/tau_1e-4/stokes_v_model/checkpoints/DeepVel_torch_epoch_148_0.36255.pt"
    ]

checkpoints = checkpoints_V
taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
# model_names = ["Stokes_I"]
model_names = ["Stokes_V"]
version = "v1"
model_paths = ["/home/xenoss/dat/thesis/models/"]
model_types = [ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V, ModelType.STOKES_V]
input_shape = (2, 43, 64, 64)
# stokes_profile = "I"
stokes_profile = "V"
#csvs = ["stokes_I_last2_tau_1.0", "stokes_I_last2_tau_1e-1", "stokes_I_last2_tau_1e-2", "stokes_I_last2_tau_1e-3", "stokes_I_last2_tau_1e-4"]
csvs = ["stokes_V_last2_tau_1.0", "stokes_V_last2_tau_1e-1", "stokes_V_last2_tau_1e-2", "stokes_V_last2_tau_1e-3", "stokes_V_last2_tau_1e-4"]

# for i, ckpt in enumerate(checkpoints):

#     deepvel_net = DeepVel_run(root = main_root, tau = taus[i], in_shape = input_shape, batch = 8, dataset_path = dataset_path, network_path = f"/home/xenoss/dat/thesis/models/v1/tau_{taus[i]}/stokes_{stokes_profile.lower()}_model/checkpoints/", stokes_profile=stokes_profile)
#     params_model = ckpt

#     if model_types[i] == ModelType.STOKES_I:
#         input_path = test_path + f"inputs/stokes_I/"

#     elif model_types[i] == ModelType.STOKES_V:
#         input_path = test_path + f"inputs/stokes_V/"

#     input_test_data = os.listdir(input_path)
#     input_test_data.sort()
#     vel_path = test_path + f"labels/tau_{taus[i]}/"
#     vel_test_data = os.listdir(vel_path)
#     vel_test_data.sort()

#     metrics_rows = []
#     print(f"Evaluating model for tau = {taus[i]} ...")
#     for j in range(len(input_test_data)):
#         input = torch.from_numpy(np.load(os.path.join(input_path, input_test_data[j])).astype(np.float32))
#         ground_truth = torch.from_numpy(np.load(os.path.join(vel_path, vel_test_data[j])).astype(np.float32))[0]
#         print("GT shape:", ground_truth.shape)
#         prediction = deepvel_net.predict(input,params_model)    

#         divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
#         divergence_pred = get_divergence(prediction[0], prediction[1])
#         vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
#         vorticity_pred = get_vorticity(prediction[0], prediction[1])

#         mse_l, rmse_l, pearson0, pearson1, pearson2, slope_x, slope_y, slope_z = get_all_metrics(ground_truth, prediction).values()
#         mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
#         mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

#         row = {
#             'tau': taus[i],
#             'sample_index': j,
#             'mse_vel': mse_l,
#             'rmse_vel': rmse_l,
#             'pearson_vel_x': pearson0,
#             'pearson_vel_y': pearson1,
#             'pearson_vel_z': pearson2,
#             'slope_vel_x': slope_x,
#             'slope_vel_y': slope_y,
#             'slope_vel_z': slope_z,
#             'mse_div': mse_d,
#             'rmse_div': rmse_d,
#             'pearson_div': pearson_d,
#             'slope_div': slope_d,
#             'mse_vort': mse_v,
#             'rmse_vort': rmse_v,
#             'pearson_vort': pearson_v,
#             'slope_vort': slope_v
#         }
#         metrics_rows.append(row)

#     csv_dir = os.path.join(main_root, f'models/{version}/test_metrics/')

#     df_tau = pd.DataFrame(metrics_rows)
#     avg_series = df_tau.select_dtypes(include=[np.number]).mean()
#     avg_row = avg_series.to_dict()
#     avg_row['tau'] = taus[i]
#     avg_row['sample_index'] = 'AVERAGE'
#     df_tau = pd.concat([df_tau, pd.DataFrame([avg_row])], ignore_index=True)

#     csv_path_tau = os.path.join(csv_dir, f'{csvs[i]}_metrics_tau_{taus[i]}.csv')
#     df_tau.to_csv(csv_path_tau, index=False)

#######################################################################################

########## NOTE PLOTTING TEST OF STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################
test_path = "/home/xenoss/dat/thesis/data/v1/dataset/test/last_2_layers/"
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

for i, tau in enumerate(taus):

    model_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/checkpoints/"
    test_save_path = f"/home/xenoss/dat/thesis/models/{version}/tau_{taus[i]}/{(model_names[0]).lower()}_model/test/"
    deepvel_net = DeepVel_run(root = main_root, in_shape=input_shape, batch = 8, dataset_path = dataset_path, network_path = model_path, tau=taus[i], stokes_profile=stokes_profile)
    params_model = checkpoints[i]
    print(f"Evaluating model for tau = {taus[i]} ...")
    plot_prediction_and_scatter_full_map_vertical(deepvel_object = deepvel_net, params_model = params_model, 
                dataset_path = test_path, test_save_path = test_save_path, 
                name = f"{model_names[0]}_{taus[i]}", zoomed_in_size = None, 
                plot_intensity = False, n_input_channels = 2, scatter_dim = None, zoom_out = 0, 
                write_metrics = False,denormalized=True, mean_vel=[mean[taus[i]]], std_vel=[std[taus[i]]], return_metrics = False, model_type = model_types[i], 
                title = model_names[0]+", tau = " + taus[i], tau = taus[i])
    
#######################################################################################