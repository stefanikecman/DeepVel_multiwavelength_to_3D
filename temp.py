import numpy as np
import muram as muram
import matplotlib.pyplot as plt

#STOKES I, STOKES V, STOKES IV HYBRID
losses = ["/home/xenoss/dat/thesis/models/models_v1/stokes_i_model/checkpoints/deepvel_torch_train_params_12_02_2025_05_08_20.npy", "/home/xenoss/dat/thesis/models/models_v1/stokes_v_model/checkpoints/deepvel_torch_train_params_12_03_2025_06_47_02.npy", "/home/xenoss/dat/thesis/models/models_v1/hybrid_model/checkpoints/deepvel_torch_train_params_12_06_2025_15_22_52.npy"]
name = ["Stokes I model", "Stokes V model"]
train_losses = []
val_losses = []

plt.figure(figsize=(10, 6))
plt.rcParams['font.size'] = 17

for i in range(len(losses)):
    with open(losses[i], "rb") as f:
        params_dict = np.load(f, allow_pickle=True).item()
        save_losses = np.load(f, allow_pickle=True)

    x = save_losses[:,0].tolist()
    train_losses.append(np.sqrt(save_losses[:,1].tolist()))
    val_losses.append(np.sqrt(save_losses[:,2].tolist()))

    
    plt.plot(x, train_losses[i], label=f'{name[i]} train loss')
    plt.plot(x, val_losses[i], label=f'{name[i]} val loss')
    #plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (RMSE)')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.savefig(f'/home/xenoss/dat/thesis/models/models_v1/training_validation_all_losses.png')

# with open("/home/xenoss/dat/thesis/models/models_v1/hybrid_model/checkpoints/deepvel_torch_train_params_12_06_2025_15_22_52.npy", "rb") as f:
#     params_dict = np.load(f, allow_pickle=True).item()
#     save_losses = np.load(f, allow_pickle=True)

# #print(save_losses)
# print (save_losses.shape)

# x = save_losses[:,0].tolist()
# train_loss = np.sqrt(save_losses[:,1].tolist())
# val_loss = np.sqrt(save_losses[:,2].tolist())

# plt.figure(figsize=(10, 6))
# plt.rcParams['font.size'] = 17
# plt.plot(x, train_loss, label='train loss')
# plt.plot(x, val_loss, label='val loss')
# #plt.yscale('log')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (RMSE)')
# plt.title('Hybrid Stokes IV model - Training and Validation Loss Curves')
# plt.legend()
# #plt.grid(True)
# plt.savefig('/home/xenoss/dat/thesis/models/models_v1/hybrid_model/training_validation_loss_hybrid.png')
# # plt.show()