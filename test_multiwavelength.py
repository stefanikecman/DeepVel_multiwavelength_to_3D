from metrics_and_plotting import plot_prediction_and_scatter_full_map_vertical
from deepvel_torch import DeepVel_run
from hybrid_deepvel_torch import DeepVel_run as DeepVel_run_hybrid
from metrics_and_plotting import ModelType

if (__name__ == '__main__'):

    main_root = "/dat/xenoss/"
    
    # deepvel_net = DeepVel_run(root = main_root, in_shape=(2,43,64,64), batch = 8, dataset_path = '/home/xenoss/dat/thesis/dataset/dataset_v1/', network_path = "/home/xenoss/dat/thesis/models/models_v1/stokes_i_model/checkpoints/")
    # params_model = '/home/xenoss/dat/thesis/models/models_v1/stokes_i_model/checkpoints/DeepVel_torch_epoch_75_0.04212.pt'
    # metrics = plot_prediction_and_scatter_full_map_vertical(deepvel_net, params_model, dataset_path='/home/xenoss/dat/thesis/dataset/dataset_v1/test/', test_save_path='/home/xenoss/dat/thesis/models/models_v1/stokes_i_model/test/', 
    #                                               name = "stokes_I_v1_test", n_input_channels=2, write_metrics=False, model_type=ModelType.STOKES_I, return_metrics=True, title= "Stokes I model - dataset v1 test", plot_intensity=False)
    
    # print(metrics)

    # deepvel_net = DeepVel_run(root = main_root, in_shape=(2,43,64,64), batch = 8, dataset_path = '/home/xenoss/dat/thesis/dataset/dataset_v1/', network_path = "/home/xenoss/dat/thesis/models/models_v1/stokes_v_model/checkpoints/")
    # params_model = '/home/xenoss/dat/thesis/models/models_v1/stokes_v_model/checkpoints/DeepVel_torch_epoch_112_0.07866.pt'
    # metrics = plot_prediction_and_scatter_full_map_vertical(deepvel_net, params_model, dataset_path='/home/xenoss/dat/thesis/dataset/dataset_v1/test/', test_save_path='/home/xenoss/dat/thesis/models/models_v1/stokes_v_model/test/', 
    #                                               name = "stokes_V_v1_test", n_input_channels=2, write_metrics=False, model_type=ModelType.STOKES_V, return_metrics=True, title= "Stokes V model - dataset v1 test", plot_intensity=False)
    
    # print(metrics)

    deepvel_net = DeepVel_run_hybrid(root = main_root, in_shape=(2,43,64,64), batch = 8, dataset_path = '/home/xenoss/dat/thesis/dataset/dataset_v1/', network_path = "/home/xenoss/dat/thesis/models/models_v1/hybrid_model/checkpoints/")
    params_model = '/home/xenoss/dat/thesis/models/models_v1/hybrid_model/checkpoints/DeepVel_torch_epoch_112_0.03388.pt'
    metrics = plot_prediction_and_scatter_full_map_vertical(deepvel_net, params_model, dataset_path='/home/xenoss/dat/thesis/dataset/dataset_v1/test/', test_save_path='/home/xenoss/dat/thesis/models/models_v1/hybrid_model/test/', 
                                                  name = "hybrid_stokes_IV_v1_test", n_input_channels=2, write_metrics=False, model_type=ModelType.STOKES_IV, return_metrics=True, title= "Hybrid Stokes IV model - dataset v1 test", plot_intensity=False)
    
    print(metrics)