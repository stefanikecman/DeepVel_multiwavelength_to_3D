import numpy as np
import muram as muram
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from analysis_fn import get_divergence, get_vorticity


# path = "/dat/milic/"
# B_unit=np.sqrt(4*np.pi)
# iter=16500
# data = muram.MuramTauSlice(path,iter,1.0)
# print(type(data.vz.T/1e5))

with open("/home/xenoss/dat/thesis/models/models_v1/hybrid_model/checkpoints/deepvel_torch_train_params_12_06_2025_15_22_52.npy", "rb") as f:
    params_dict = np.load(f, allow_pickle=True).item()
    save_losses = np.load(f, allow_pickle=True)

#print(save_losses)
print (save_losses.shape)

x = save_losses[:,0].tolist()
train_loss = np.sqrt(save_losses[:,1].tolist())
val_loss = np.sqrt(save_losses[:,2].tolist())

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 17
plt.plot(x, train_loss, label='train loss')
plt.plot(x, val_loss, label='val loss')
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.title('Hybrid Stokes IV model - Training and Validation Loss Curves')
plt.legend()
#plt.grid(True)
plt.savefig('/home/xenoss/dat/thesis/models/models_v1/hybrid_model/training_validation_loss_hybrid.png')
# plt.show()

# data_cube_vx = np.load("/home/xenoss/dat/datasets_5x5_experiments/main_dataset_train/stacked_velocities_x_normalized.npy")
# data_cube_vy = np.load("/home/xenoss/dat/datasets_5x5_experiments/main_dataset_train/stacked_velocities_y_normalized.npy")

# divergence = get_divergence(data_cube_vx, data_cube_vy).cpu().numpy()
# vorticity = get_vorticity(data_cube_vx, data_cube_vy).cpu().numpy()

# mean_div = np.mean(divergence)
# std_div = np.std(divergence)
# mean_vor = np.mean(vorticity)
# std_vor = np.std(vorticity)
# print("Divergence:")
# print("mean:", mean_div)
# print("std:", std_div)
# print("Vorticity:")
# print("mean:", mean_vor)
# print("std:", std_vor)

# mse1 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse1.npy")
# mse2 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse2.npy")
# mse3 = np.load("/home/xenoss/data/kecman_project/DeepVel_3D_velocity/new_loss_experiments/version_9/mse3.npy")

# def average_chunks(arr, chunk_size):
#     return np.mean(arr.reshape(-1, chunk_size), axis=1)

# mse1 = average_chunks(mse1, 862)
# mse2 = average_chunks(mse2, 862)
# mse3 = average_chunks(mse3, 862)

# x = list(range(0, 120))

# plt.figure(figsize=(10, 6))
# plt.rcParams['font.size'] = 17
# plt.plot(x, mse1, label='MSE 1')
# plt.plot(x, mse2/10, label='MSE 2')
# plt.plot(x, mse3/10, label='MSE 3')
# #plt.yscale('log')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (MSE)')
# plt.title('Individual Loss Components ')
# plt.legend()
# #plt.grid(True)
# plt.savefig('new_loss_curves_v9.png')
# plt.show()