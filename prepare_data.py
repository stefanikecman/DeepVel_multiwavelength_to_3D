import os
import shutil
import muram as muram
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

main_path = "/dat/milic/2D/"
destination_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/intensities/"
destination_velocities = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities/"
dataset_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset/"

def load_data (origin_path, destination_path):
    filenames = os.listdir(origin_path)
    intensities = []
    for n in filenames:
        if "I_out" in n:
            intensities.append(n)

    for i in intensities:
        shutil.copy(origin_path + i, destination_path)

def file_naming(name, iter):
    padding = ""
    if iter <10:
        padding = "00000"
    elif iter<100 :
        padding = "0000"
    elif iter<1000:
        padding = "000"
    elif iter<10000:
        padding = "00"
    else: padding = "0"
    return name + "_" + padding + str(iter) 

def create_velocities(main_path, velocity_path, tau = 1.0): #could have done with .npz but realised it too late

    #should I divide the velocities with 1e5 now to get it in km/h?
    #also I am thinking about adding the extent later in the plotting

    for i in range(0, 18050, 50):
        data = muram.MuramTauSlice(main_path,i,1.0)
        vx = data.vy
        vy = data.vz
        np.save(velocity_path+file_naming("vx", i)+".npy", vx)
        np.save(velocity_path+file_naming("vy", i)+".npy", vy)

def create_dataset(dataset_path, int_path, vel_path):
    indices1 = list(range(0, 18000, 50))
    indices2 = list(range(50, 18050, 50))

    intensities = os.listdir(int_path)
    velocities = os.listdir(vel_path)

    vxs = [v for v in velocities if "vx" in v]
    vys = [v for v in velocities if "vy" in v]

#    count = 0
#    for i in range(len(indices1)):
#        i1 = muram.MuramIntensity(int_path, indices1[i])
#        i2 = muram.MuramIntensity(int_path, indices2[i])

#        intensities_concat = np.stack([i1, i2], axis = 0)
#        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
#        count +=1
    
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices1[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices2[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices1[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices2[i])+ ".npy")

        vx_out = (vx1 + vx2) / 2
        vy_out = (vy1 + vy2) / 2

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        count+=1


def crop_image_center (original_dir, input_data_name, new_dim, save_dir):

    input_data = np.load(original_dir + input_data_name)

    _, h, w = input_data.shape
    cropped_img = []

    y_im_center = int(h/2)
    x_im_center = int(w/2)

    cropped_img = input_data[:, (y_im_center - int(new_dim/2)): (y_im_center + int(new_dim/2)), (x_im_center - int(new_dim/2)): (x_im_center + int(new_dim/2))]

    np.save(save_dir + input_data_name.replace(".npy", "_cropped.npy"), cropped_img)


def crop_image (original_dir, input_data_name, new_dim, save_dir):

    input_data = np.load(original_dir + input_data_name)

    _, h, w = input_data.shape
    cropped_img = []

    for i in range(0, h, new_dim):
        for j in range(0, w, new_dim):
            cropped_img.append(input_data[:, i: (i + new_dim), j: (j + new_dim)])
    
    for i in range(len(cropped_img)):
        np.save(save_dir + input_data_name.replace(".npy", "_") + str (i) + ".npy", cropped_img[i])

def normalize_layerwise (input_data):
    #Stefani added

    if (torch.is_tensor(input_data)):
        mean_0 = torch.mean(input_data[0])
        mean_1 = torch.mean(input_data[1])
        std_0 = torch.std(input_data[0])
        std_1 = torch.std(input_data[1])

        norm_input_data = input_data.detach().clone()

        norm_input_data [0] = (input_data[0] - mean_0) / std_0
        norm_input_data [1] = (input_data[1] - mean_1) / std_1


    else:
        mean_0 = np.mean(input_data[0])
        mean_1 = np.mean(input_data[1])

        std_0 = np.std(input_data[0])
        std_1 = np.std(input_data[1])

        norm_input_data = input_data.copy()

        norm_input_data [0] = (input_data[0] - mean_0) / std_0
        norm_input_data [1] = (input_data[1] - mean_1) / std_1

    return norm_input_data


def plot_predictions(int_gt, vel_gt, vel_pred, filename):

    matplotlib.use('agg')
    plt.figure(figsize = (20, 80))
    fig, ax = plt.subplots(nrows = 3, ncols = 2, sharex = True, sharey = True)

    intensity_0 = int_gt[0, :, :].cpu().numpy()
    intensity_1 = vel_gt[1, :, :].cpu().numpy()

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

    plt.savefig('test_results/'+filename + '.png')


#load_data (main_path, destination_path)
#create_velocities(main_path, destination_velocities, 1.0)
#create_dataset(dataset_path, destination_path, destination_velocities)


input_dir = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset/inputs/"
labels_dir = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset/labels/"

cropped_dir = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset_cropped_center/"

intensities = os.listdir(input_dir)
velocities = os.listdir(labels_dir)


#for i in intensities:
#    crop_image_center(input_dir, i, 64, cropped_dir+"inputs/")


#for v in velocities:
#   crop_image_center(labels_dir, v, 64, cropped_dir+"labels/")


im = np.load('/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset/inputs/intensities_000344.npy')
print(im.shape)
