from curses import version
import numpy as np
import torch
import os
import csv
from collections import OrderedDict
from multiheight_deepvel_torch import DeepVel_run
import pandas as pd
from analysis_fn import get_all_metrics, get_divergence, get_vorticity
from metrics_and_plotting import ModelType, plot_prediction_and_scatter_full_map_vertical,plot_prediction_and_scatter_full_map_vertical_multiheight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_root = "/home/xenoss/dat/thesis/"


metrics_by_tau = OrderedDict()

########## NOTE TESTING STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################
version = "v12"
checkpoints = [f"/home/xenoss/dat/thesis/models/{version}/stokes_i_model/checkpoints/DeepVel_torch_epoch_93_0.26813.pt"]
model_names = ["Stokes_I"]

dataset_path = f'/home/xenoss/dat/thesis/data/{version}/dataset/train/'
test_path = f'/home/xenoss/dat/thesis/data/{version}/dataset/test/last_2_layers/'
model_paths = ["/home/xenoss/dat/thesis/models/"]
model_types = [ModelType.STOKES_I]
input_shape = (2, 156, 64, 64)
output_shape = (5, 3, 64, 64)
stokes_profile = "I"
csvs = [f"multiheight_stokes_{stokes_profile.lower()}_last2"]
common_csv = []
taus = ["1.0", "1e-1", "1e-2", "1e-3", "1e-4"]
for i, ckpt in enumerate(checkpoints):

    deepvel_net = DeepVel_run(root = main_root, in_shape = input_shape, out_shape = output_shape, batch = 1, dataset_path = dataset_path, 
                              network_path = f"/home/xenoss/dat/thesis/models/{version}/stokes_{stokes_profile.lower()}_model/checkpoints/", 
                              stokes_profile=stokes_profile, test=True)
        
    params_model = ckpt

    if model_types[0] == ModelType.STOKES_I:
        input_path = test_path + f"inputs/stokes_I/"

    elif model_types[0] == ModelType.STOKES_V:
        input_path = test_path + f"inputs/stokes_V/"
    

    input_test_data = os.listdir(input_path)
    input_test_data.sort()

    metrics_rows = []
    for j in range(len(input_test_data)):
        input = torch.from_numpy(np.load(os.path.join(input_path, input_test_data[j])).astype(np.float32)).unsqueeze(0)
        # print("Input shape:", input.shape)
        prediction = deepvel_net.predict(input,params_model)  
        # print("Prediction shape:", prediction.shape)  

        for k in range(len(taus)):
            print(f"Metrics for tau = {taus[k]}:")
            vel_path = test_path + f"labels/tau_{taus[k]}/"
            vel_test_data = os.listdir(vel_path)
            vel_test_data.sort()
            
            ground_truth = torch.from_numpy(np.load(os.path.join(vel_path, vel_test_data[j])).astype(np.float32))[0]
            prediction_tau = prediction[k, :, :, :]
            # print("GT shape:", ground_truth.shape)

            divergence_orig = get_divergence(ground_truth[0], ground_truth[1])
            divergence_pred = get_divergence(prediction_tau[0], prediction_tau[1])
            vorticity_orig = get_vorticity(ground_truth[0], ground_truth[1])
            vorticity_pred = get_vorticity(prediction_tau[0], prediction_tau[1])

            mse_l, rmse_l, pearson0, pearson1, pearson2, slope_x, slope_y, slope_z = get_all_metrics(ground_truth, prediction_tau).values()
            mse_d, rmse_d, pearson_d, slope_d = get_all_metrics(divergence_orig, divergence_pred, is_divergence=True).values()
            mse_v, rmse_v, pearson_v, slope_v = get_all_metrics(vorticity_orig, vorticity_pred, is_divergence=True).values()

            row = {
                'tau': taus[k],
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
    csv_path_tau = os.path.join(csv_dir, f'{csvs[i]}_metrics.csv')
    df_tau.to_csv(csv_path_tau, index=False)
#######################################################################################

########## NOTE PLOTTING TEST OF STOKES MODELS ON LAST 2 LAYERS OF SIMULATION ################

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

model_path = f"/home/xenoss/dat/thesis/models/{version}/{(model_names[0]).lower()}_model/checkpoints/"
test_save_path = f"/home/xenoss/dat/thesis/models/{version}/{(model_names[0]).lower()}_model/test/"


deepvel_net = DeepVel_run(root = main_root, in_shape=input_shape, out_shape=output_shape, batch = 1, dataset_path = dataset_path, network_path = model_path, 
                              stokes_profile=stokes_profile, test=True)
params_model = checkpoints[0]

plot_prediction_and_scatter_full_map_vertical_multiheight(deepvel_object = deepvel_net, params_model = params_model, 
                dataset_path = test_path, test_save_path = test_save_path, 
                name = f"{model_names[0]}", zoomed_in_size = None, 
                input_shape = input_shape, output_shape=output_shape, scatter_dim = None, zoom_out = 0, 
                write_metrics = False,denormalized=True, mean_vels=mean, std_vels=std, return_metrics = False, model_type = model_types[0], 
                title = model_names[0], taus = taus)

plot_prediction_and_scatter_full_map_vertical_multiheight(deepvel_object = deepvel_net, params_model = params_model, 
                dataset_path = test_path, test_save_path = test_save_path, 
                name = f"{model_names[0]}", zoomed_in_size = None, 
                input_shape = input_shape, output_shape=output_shape, scatter_dim = None, zoom_out = 0, 
                write_metrics = False,denormalized=False, mean_vels=mean, std_vels=std, return_metrics = False, model_type = model_types[0], 
                title = model_names[0], taus = taus)   

    
#######################################################################################