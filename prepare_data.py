import os
import shutil
import muram as muram
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_path = "/dat/milic/2D/"
intensities_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/intensities/"
#destination_velocities = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities/"
velocities_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities_normalized_together/"
dataset_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/main_dataset_experiment2/"

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

def create_dataset_2_intensities (dataset_path, int_path, vel_path):
    indices1 = list(range(0, 18000, 50))
    indices2 = list(range(50, 18050, 50))

    velocities = os.listdir(vel_path)
    velocities.sort()

    #print(velocities [0])
    #print (velocities[1])

    
    vxs = [v for v in velocities if "vx" in v]
    vys = [v for v in velocities if "vy" in v]
    #print(vxs[0])
    #print(vys[0])

    
    count = 0
    for i in range(len(indices1)):
        i1 = muram.MuramIntensity(int_path, indices1[i])
        i2 = muram.MuramIntensity(int_path, indices2[i])
        #print('I1 has index ', indices1[i])
        #print('I2 has index ', indices2[i])
        
        intensities_concat = np.stack([i1, i2], axis = 0)

        #plt.figure(figsize=[8,3])
        #plt.subplot(121)
        #plt.imshow(intensities_concat[0,1200:1264, 500:564].T, origin='lower')
        #plt.subplot(122)
        #plt.imshow(intensities_concat[0,1200:1264, 500:564].T, origin='lower')
        #plt.savefig('debug_intensities.png')

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1

    
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices1[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices2[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices1[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices2[i])+ ".npy")

        #print('vx1 has index ', file_naming("vx", indices1[i]))
        #print('vy1 has index ', file_naming("vy", indices1[i]))
        #print('vx2 has index ', file_naming("vx", indices2[i]))
        #print('vy2 has index ', file_naming("vy", indices2[i]))

        vx_out = (vx1 + vx2) / 2
        vy_out = (vy1 + vy2) / 2

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        count+=1

def create_dataset_4_intensities (dataset_path, int_path, vel_path):
    indices1 = list(range(0, 17900, 50))
    indices2 = list(range(50, 17950, 50)) # also corresponds to v
    indices3 = list(range(100, 18000, 50)) # also corresponds to v
    indices4 = list(range(150, 18050, 50))

    velocities = os.listdir(vel_path)
    velocities.sort()

    #print(velocities [0])
    #print (velocities[1])

    
    vxs = [v for v in velocities if "vx" in v]
    vys = [v for v in velocities if "vy" in v]
    #print(vxs[0])
    #print(vys[0])

    
    count = 0
    for i in range(len(indices1)):
        
        i1 = muram.MuramIntensity(int_path, indices1[i])
        i2 = muram.MuramIntensity(int_path, indices2[i])
        i3 = muram.MuramIntensity(int_path, indices3[i])
        i4 = muram.MuramIntensity(int_path, indices4[i])
        
        intensities_concat = np.stack([i1, i2, i3, i4], axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
        

    
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices2[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices3[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices2[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices3[i])+ ".npy")


        
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

def crop_image_every_nth (n, original_dir, input_data_name, new_dim, save_dir):

    input_data = np.load(original_dir + input_data_name)

    _, h, w = input_data.shape
    cropped_img = []

    for i in range(0, h, new_dim):
        for j in range(0, w, new_dim):
            cropped_img.append(input_data[:, i: (i + new_dim), j: (j + new_dim)])
    
    for i in range(0, len(cropped_img), n):
        np.save(save_dir + input_data_name.replace(".npy", "_cropped_") + str (i) + ".npy", cropped_img[i])


def normalize_layerwise_old(input_data):
    #Stefani added
    '''
    Normalizes each datapoint separately, layerwise
    '''

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

def normalize_layerwise (input_data):
    '''
    Normalizes each datapoint separately, layers all together
    '''
    if (torch.is_tensor(input_data)):
        mean = torch.mean(input_data)
        std = torch.std(input_data)
        norm_input_data = input_data.detach().clone()

    else:
        mean = np.mean(input_data)
        std = np.std(input_data)

        norm_input_data = input_data.copy()

    norm_input_data = (input_data - mean) / std

    return norm_input_data

def normalize_all_outputs (output_path, save_path):
    '''
    Normalizes outputs (velocities) all together
    '''
    velocity_names = os.listdir(output_path)
    velocity_names.sort()

    velocities = [np.load(output_path + v) for v in velocity_names]
    mean = np.mean(velocities)
    std = np.std(velocities)

    for v in velocity_names:
        velocity = np.load(output_path + v)
        velocity = (velocity - mean)/std
        np.save(save_path + v, velocity)
    
    return mean, std




if (__name__ == '__main__'):

    '''
    vel_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities/"
    save_normalized_velocities = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities_normalized_together/"

    mean, std = normalize_all_outputs(vel_path, save_normalized_velocities)
    print(f"Mean is: {mean}, std is: {std}")
    '''
    
    #load_data (main_path, destination_path)
    #create_velocities(main_path, destination_velocities, 1.0)
    #create_dataset_4_intensities(dataset_path, intensities_path, velocities_path)

    '''
    input_dir = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/main_dataset_experiment2/inputs/"
    labels_dir = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/main_dataset_experiment2/labels/"

    
    cropped_dir_68 = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset_cropped_96x96/dataset_cropped_68k_experiment2/"
    cropped_dir_34 = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/dataset_cropped_96x96/dataset_cropped_34k_experiment2/"
    
    intensities = os.listdir(input_dir)
    velocities = os.listdir(labels_dir)

    intensities.sort()
    velocities.sort()
    '''
    #for i in intensities:
    #    crop_image_every_third(input_dir, i, 96,  cropped_dir_68+"inputs/")
    #    crop_image_every_sixth(input_dir, i, 96, cropped_dir_34+"inputs/")
        

    '''
    for v in velocities:
        crop_image_every_third(labels_dir, v, 96,  cropped_dir_68+"labels/")
        crop_image_every_sixth(labels_dir, v, 96, cropped_dir_34+"labels/")      

    im = np.load('/home/xenoss/data/kecman_project/DeepVel_3D_velocity/main_dataset_experiment2/labels/velocities_000000.npy')
    print(im.shape)
    '''