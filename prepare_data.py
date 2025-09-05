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
import random
import shutil
import muram as mio

main_path = "/dat/milic/2D/"
intensities_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/intensities/"
destination_velocities = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities/"
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

def create_B (main_path, B_path_save, tau = 1.0):

    for i in range(0, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,1.0)
        B = data.Bx.T # or dont transpose?
        np.save(B_path_save+file_naming("B", i)+".npy", B)

def create_vz (main_path, vz_path_save, tau = 1.0):

    for i in range(0, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,1.0)
        vz = data.vz.T # or dont transpose?
        np.save(vz_path_save+file_naming("vz", i)+".npy", vz)

def create_dataset_2_intensities (dataset_path, int_path, vel_path, last_idx):
    indices1 = list(range(0, last_idx, 50))
    indices2 = list(range(50, last_idx + 50, 50))

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

def create_dataset_4_intensities (dataset_path, int_path, vel_path, last_idx):
    indices1 = list(range(0, last_idx-100, 50))
    indices2 = list(range(50, last_idx-50, 50)) # also corresponds to v
    indices3 = list(range(100, last_idx, 50)) # also corresponds to v
    indices4 = list(range(150, last_idx+50, 50))

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

def create_dataset_6_intensities (dataset_path, int_path, vel_path, last_idx):
    indices1 = list(range(0, last_idx - 200, 50))
    indices2 = list(range(50, last_idx - 150, 50)) 
    indices3 = list(range(100, last_idx - 100, 50)) # also corresponds to v
    indices4 = list(range(150, last_idx - 50, 50)) # also corresponds to v
    indices5 = list(range(200, last_idx, 50)) 
    indices6 = list(range(250, last_idx+50, 50))

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
        i5 = muram.MuramIntensity(int_path, indices5[i])
        i6 = muram.MuramIntensity(int_path, indices6[i])
        
        intensities_concat = np.stack([i1, i2, i3, i4, i5, i6], axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
        

    
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices3[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices4[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices3[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices4[i])+ ".npy")


        
        vx_out = (vx1 + vx2) / 2
        vy_out = (vy1 + vy2) / 2

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        
        count+=1

def create_dataset_8_intensities (dataset_path, int_path, vel_path, last_idx):
    indices1 = list(range(0, last_idx - 300, 50))
    indices2 = list(range(50, last_idx - 250, 50)) 
    indices3 = list(range(100, last_idx - 200, 50))
    indices4 = list(range(150, last_idx - 150, 50)) # also corresponds to v
    indices5 = list(range(200, last_idx - 100, 50)) # also corresponds to v
    indices6 = list(range(250, last_idx - 50, 50)) 
    indices7 = list(range(300, last_idx, 50)) 
    indices8 = list(range(350, last_idx+50, 50))

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
        i5 = muram.MuramIntensity(int_path, indices5[i])
        i6 = muram.MuramIntensity(int_path, indices6[i])
        i7 = muram.MuramIntensity(int_path, indices7[i])
        i8 = muram.MuramIntensity(int_path, indices8[i])
        
        intensities_concat = np.stack([i1, i2, i3, i4, i5, i6, i7, i8], axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
        

    
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices4[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices5[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices4[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices5[i])+ ".npy")


        
        vx_out = (vx1 + vx2) / 2
        vy_out = (vy1 + vy2) / 2

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        
        count+=1

def create_dataset_10_intensities (dataset_path, int_path, vel_path, last_idx):
    indices1 = list(range(0, last_idx - 400, 50))
    indices2 = list(range(50, last_idx - 350, 50)) 
    indices3 = list(range(100, last_idx - 300, 50))
    indices4 = list(range(150, last_idx - 250, 50)) 
    indices5 = list(range(200, last_idx - 200, 50)) # also corresponds to v
    indices6 = list(range(250, last_idx - 150, 50)) # also corresponds to v
    indices7 = list(range(300, last_idx-100, 50)) 
    indices8 = list(range(350, last_idx-50, 50))
    indices9 = list(range(400, last_idx, 50)) 
    indices10 = list(range(450, last_idx+50, 50))

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
        i5 = muram.MuramIntensity(int_path, indices5[i])
        i6 = muram.MuramIntensity(int_path, indices6[i])
        i7 = muram.MuramIntensity(int_path, indices7[i])
        i8 = muram.MuramIntensity(int_path, indices8[i])
        i9 = muram.MuramIntensity(int_path, indices9[i])
        i10 = muram.MuramIntensity(int_path, indices10[i])

        intensities_concat = np.stack([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10], axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
        
    count = 0
    for i in range(len(indices1)):
        vx1 = np.load(vel_path + file_naming("vx", indices5[i])+ ".npy")
        vx2 = np.load(vel_path + file_naming("vx", indices6[i])+ ".npy")

        vy1 = np.load(vel_path + file_naming("vy", indices5[i])+ ".npy")
        vy2 = np.load(vel_path + file_naming("vy", indices6[i])+ ".npy")


        
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

def crop_image_every_nth (n, original_dir, data_idx, new_dim, save_dir, sample_random = False):

    input_data = np.load(original_dir + 'inputs/' + 'intensities_' + data_idx + '.npy')
    label_data = np.load(original_dir + 'labels/' + 'velocities_' + data_idx + '.npy')

    _, h, w = input_data.shape
    cropped_img_inputs = []
    cropped_img_labels = []

    for i in range(0, h, new_dim):
        for j in range(0, w, new_dim):
            cropped_img_inputs.append(input_data[:, i: (i + new_dim), j: (j + new_dim)])
            cropped_img_labels.append(label_data[:, i: (i + new_dim), j: (j + new_dim)])
    
    if sample_random == False: # sample uniformly
        for i in range(0, len(cropped_img), n):
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])
    
    else: #sample randomly
        to_save = int(len(cropped_img_inputs) / n)
        indices_to_save = random.sample(range(0, len(cropped_img_inputs)), to_save)

        for i in indices_to_save:
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])



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

def denormalize_data (data, mean, std):
    return data*std + mean

def generate_data_5x5_experiments(n_samples, intensities, vx, vy):
    '''
    Generates data for 5x5 experiments, where each intensity has two velocities
    '''
    #TODO UNCOMMENT THIS TO GAVE THE ORIGINAL FUNCTION WITH RANDOM LOGIC
    
    # n_random = int(n_samples**(1/3))  # number of random samples to be generated
    # center_h = random.sample(range(64, 1474), n_random)
    # center_w = random.sample(range(64, 1474), n_random)
    # center_c = random.sample(range(5, 237), n_random)

    # with open('/home/xenoss/data/kecman_project/DeepVel_3D_velocity/centers.txt', 'w') as f:
    #     f.write("center_c: " + ", ".join(map(str, center_c)) + "\n")
    #     f.write("center_h: " + ", ".join(map(str, center_h)) + "\n")
    #     f.write("center_w: " + ", ".join(map(str, center_w)) + "\n")

    center_c = [194, 226, 214, 215, 147, 228, 134, 183, 171, 96, 170, 178, 11, 113, 70, 49, 127, 185, 181, 198, 210, 61, 40, 53, 153, 165, 29, 230, 219, 42, 141, 46, 112, 23, 148, 8, 85, 91, 28, 111, 109]
    center_h = [208, 101, 941, 1163, 1006, 508, 745, 613, 1203, 527, 983, 1389, 1029, 1219, 879, 116, 898, 1275, 205, 952, 780, 506, 238, 583, 1024, 1465, 331, 772, 723, 95, 1423, 225, 471, 392, 1339, 843, 1117, 920, 82, 743, 853]
    center_w = [1189, 751, 439, 903, 1438, 821, 996, 367, 1071, 764, 69, 286, 162, 1363, 1056, 213, 816, 1130, 90, 615, 1315, 709, 272, 268, 207, 326, 515, 169, 1214, 699, 757, 1250, 1293, 746, 703, 662, 731, 1048, 115, 1446, 1110]

    channel_range = [2, 4, 6, 8, 10]
    spatial_range = [32, 48, 64, 96, 128]

    i = 0
    for ch in center_h:
        for cw in center_w:
            for cc in center_c:
                if i %100 ==0: print(f"Processing sample {i}")
                i += 1
                for c in channel_range:
                    for s in spatial_range:
                        
                        half_s = int(s/2)
                        half_c = int(c/2)
                        intensity = intensities[cc-half_c:cc+half_c, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                        velocity_x = vx[cc-half_c:cc+half_c, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                        velocity_y = vy[cc-half_c:cc+half_c, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                        #print("Shape of intensity: ", intensity.shape)
                        np.save(f'/dat/xenoss/datasets_5x5_experiments/normalized/timestep_{c}/cropped/data_{s}x{s}/inputs/intensity_{cc}_{ch}_{cw}.npy', intensity)
                        np.save(f'/dat/xenoss/datasets_5x5_experiments/normalized/timestep_{c}/cropped/data_{s}x{s}/labels_v_separated/vx_{cc}_{ch}_{cw}.npy', velocity_x)
                        np.save(f'/dat/xenoss/datasets_5x5_experiments/normalized/timestep_{c}/cropped/data_{s}x{s}/labels_v_separated/vy_{cc}_{ch}_{cw}.npy', velocity_y)


def generate_labels_5x5_experiments ():
    '''
    Generates labels for 5x5 experiments, where each intensity has two velocities
    '''
    dataset_path = "/dat/xenoss/datasets_5x5_experiments/normalized/"
    temporal_range = [2, 4, 6, 8, 10]
    spatial_range = [32, 48, 64, 96, 128]
    for t in temporal_range:
        vel_center_idx = int(t/2) -1  # center index for the temporal dimension

        for s in spatial_range:
            print(f"Processing timestep {t} and spatial size {s}")
            label_path = dataset_path + f'timestep_{t}/cropped/data_{s}x{s}/labels_v_separated/'

            velocities_x = [v for v in os.listdir(label_path) if "vx" in v]
            velocities_y = [v for v in os.listdir(label_path) if "vy" in v]
            velocities_x.sort()
            velocities_y.sort()
            # print("vx: ", velocities_x[:10])
            # print("vy: ", velocities_y[:10])
            

            for i in range(len(velocities_x)):
                if velocities_x[i].replace("vx_", "") != velocities_y[i].replace("vy_", ""):
                    raise ValueError("Velocity x and y files do not match")
                
                vx = np.load(label_path + velocities_x[i])[vel_center_idx:vel_center_idx+2]
                vy = np.load(label_path + velocities_y[i])[vel_center_idx:vel_center_idx+2]
                
                vx_out = (vx[0] + vx[1]) / 2
                vy_out = (vy[0] + vy[1]) / 2

                velocities_concat = np.stack([vx_out, vy_out], axis=0)
                #print(label_path.replace('labels_v_separated', 'labels') + 'velocities_' + velocities_x[i].replace('vx_', ''))
            
                np.save(label_path.replace('labels_v_separated', 'labels') + 'velocities_' + velocities_x[i].replace('vx_', ''), velocities_concat)
                

def create_test_data_5x5_experiments():
    main_dataset = "/dat/xenoss/datasets_5x5_experiments/main_dataset_test/"

    intensity = np.load(main_dataset + 'stacked_intensities_all.npy')
    vx = np.load(main_dataset + 'stacked_velocities_x_all.npy')
    vy = np.load(main_dataset + 'stacked_velocities_y_all.npy')

    i_mean = np.mean(intensity)
    vx_mean = np.mean(vx)
    vy_mean = np.mean(vy)

    i_std = np.std(intensity)
    vx_std = np.std(vx)
    vy_std = np.std(vy)

    intensity_norm = (intensity - i_mean) / i_std
    vx_norm = (vx - vx_mean) / vx_std
    vy_norm = (vy - vy_mean) / vy_std

    intensity_10 = intensity_norm[-10:, :, :]  
    vx_10 = vx_norm[-10:, :, :]                
    vy_10 = vy_norm[-10:, :, :] 

    intensity_2 = intensity_10[4:6, :, :]
    vx_2 = vx_10[4:6, :, :]
    vy_2 = vy_10[4:6, :, :]

    intensity_4 = intensity_10[3:7, :, :]
    vx_4 = vx_10[3:7, :, :]
    vy_4 = vy_10[3:7, :, :]

    intensity_6 = intensity_10[2:8, :, :]
    vx_6 = vx_10[2:8, :, :]
    vy_6 = vy_10[2:8, :, :]

    intensity_8 = intensity_10[1:9, :, :]
    vx_8 = vx_10[1:9, :, :]
    vy_8 = vy_10[1:9, :, :]

    #NOTE In this case, all the center velocities correspond to indices 4 and 5 of the 10-channel velocities, which are equal to the vx_2 and vy_2
    velocities = np.stack([
    (vx_2[0] + vx_2[1]) / 2,
    (vy_2[0] + vy_2[1]) / 2
    ], axis=0)

    for i in range(11):
        np.save(main_dataset + "normalized/velocities_" + str(i) + ".npy", velocities)
    # np.save(main_dataset + "normalized/intensities_2.npy", intensity_2)
    # np.save(main_dataset + "normalized/intensities_4.npy", intensity_4)
    # np.save(main_dataset + "normalized/intensities_6.npy", intensity_6)
    # np.save(main_dataset + "normalized/intensities_8.npy", intensity_8)
    # np.save(main_dataset + "normalized/intensities_10.npy", intensity_10)

    # np.save(main_dataset + "normalized/vx_2.npy", vx_2)
    # np.save(main_dataset + "normalized/vx_4.npy", vx_4)
    # np.save(main_dataset + "normalized/vx_6.npy", vx_6)
    # np.save(main_dataset + "normalized/vx_8.npy", vx_8)
    # np.save(main_dataset + "normalized/vx_10.npy", vx_10)

    # np.save(main_dataset + "normalized/vy_2.npy", vy_2)
    # np.save(main_dataset + "normalized/vy_4.npy", vy_4)
    # np.save(main_dataset + "normalized/vy_6.npy", vy_6)
    # np.save(main_dataset + "normalized/vy_8.npy", vy_8)
    # np.save(main_dataset + "normalized/vy_10.npy", vy_10)

    # print("Intensity mean: ", i_mean, " Intensity std: ", i_std)
    # print("Velocity x mean: ", vx_mean, " Velocity x std: ", vx_std)
    # print("Velocity y mean: ", vy_mean, " Velocity y std: ", vy_std)
    # np.save(main_dataset + 'stacked_intensities_normalized.npy', intensity_norm)
    # np.save(main_dataset + 'stacked_velocities_x_normalized.npy', vx_norm)
    # np.save(main_dataset + 'stacked_velocities_y_normalized.npy', vy_norm)

if (__name__ == '__main__'):
    
    main_root = "/dat/xenoss/"

    ###### NOTE CREATE B AND VZ ###########

    Vz = [s.vz for s in slices]



    ######################################
    # intensities = [muram.MuramIntensity(main_dataset + 'inputs/', i) for i in range(12000, 18050, 50)]

    # stacked_intensities = np.stack(intensities, axis = 0)
    # stacked_velocities_x = np.stack([np.load(main_dataset + 'labels/' + v) for v in vel_x], axis = 0)
    # stacked_velocities_y = np.stack([np.load(main_dataset + 'labels/' + v) for v in vel_y], axis = 0)

    # print("Shape of stacked intensities: ", stacked_intensities.shape)
    # print("Shape of stacked velocities x: ", stacked_velocities_x.shape)
    # print("Shape of stacked velocities y: ", stacked_velocities_y.shape)

    # np.save(main_dataset + '/stacked_intensities_all.npy', stacked_intensities)
    # np.save(main_dataset + '/stacked_velocities_x_all.npy', stacked_velocities_x)
    # np.save(main_dataset + '/stacked_velocities_y_all.npy', stacked_velocities_y)

    # generate_data_5x5_experiments(70000, np.load('/dat/xenoss/datasets_5x5_experiments/main_dataset_train/stacked_intensities_normalized.npy'),
    #                                np.load('/dat/xenoss/datasets_5x5_experiments/main_dataset_train/stacked_velocities_x_normalized.npy'),
    #                                np.load('/dat/xenoss/datasets_5x5_experiments/main_dataset_train/stacked_velocities_y_normalized.npy'))
