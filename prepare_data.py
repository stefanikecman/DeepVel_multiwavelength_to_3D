import os
import shutil
import muram as muram
import numpy as np
import torch
import matplotlib

import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
import muram as mio
import astropy.io.fits as fits
import scipy.interpolate as sci

matplotlib.use('agg')

main_path = "/dat/milic/2D/"
intensities_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/intensities/"
destination_velocities = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities/"
velocities_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/velocities_normalized_together/"
dataset_path = "/home/xenoss/data/kecman_project/DeepVel_3D_velocity/main_dataset_experiment2/"

def load_data (origin_path, destination_path):
    """
    Load Intensity data from the origin path and save it to the destination path.
    Args:
        origin_path: Path where the original intensity files are located
        destination_path: Path where the intensity files will be copied
    """
    filenames = os.listdir(origin_path)
    intensities = []
    for n in filenames:
        if "I_out" in n:
            intensities.append(n)

    for i in intensities:
        shutil.copy(origin_path + i, destination_path)

def file_naming(name, iter):
    """
    Create a filename with leading zeros based on the iteration number.
    """
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

def create_velocities(main_path, velocity_path, tau = 1.0): 
    """
    Create velocity files for horizontal velocity components (vx, vy) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        velocity_path: Path where the velocity files will be saved
        tau: Optical depth level to extract velocities (default is 1.0)
    """

    for i in range(0, 18050, 50):
        data = muram.MuramTauSlice(main_path,i,1.0)
        vx = data.vy
        vy = data.vz
        np.save(velocity_path+file_naming("vx", i)+".npy", vx)
        np.save(velocity_path+file_naming("vy", i)+".npy", vy)

def create_B (main_path, B_path_save, tau = 1.0):
    """
    Create vertical magnetic field files (Bz) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        B_path_save: Path where the magnetic field files will be saved
        tau: Optical depth level to extract magnetic field (default is 1.0)
    """

    for i in range(0, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,1.0)
        B = data.Bx
        np.save(B_path_save+file_naming("B", i)+".npy", B)

def create_vz (main_path, vz_path_save, tau = 1.0):
    """
    Create vertical velocity files (vz) from MURaM simulation data and save them as .npy files.
    Args:
        main_path: Path where the MURaM simulation data is located
        vz_path_save: Path where the vertical velocity files will be saved
        tau: Optical depth level to extract vertical velocity (default is 1.0)
    """

    for i in range(0, 18050, 50):
        data = mio.MuramTauSlice(main_path,i,1.0)
        vz = data.vx
        np.save(vz_path_save+file_naming("vz", i)+".npy", vz)

def create_dataset_N_intensities (dataset_path, int_path, vel_path, last_idx, indices_list):
    """
    Create a dataset, where the input intensity has N channels, from the original intensity data.
    Args:
        dataset_path: Path where the new dataset will be saved
        int_path: Path where the original intensity files are located
        vel_path: Path where the original velocity files are located
        last_idx: Last index of the intensity files to be considered
        indices_list: List of lists, where each sublist contains the relative indices for the N channels
    """
    velocities = os.listdir(vel_path)
    velocities.sort()
    
    count = 0
    for i in range(len(indices_list[0])):
        intensities_concat = []
        for j in range(len(indices_list)):
            idx = indices_list[j][i]
            intensity = muram.MuramIntensity(int_path, idx)
            intensities_concat.append(intensity)
        
        intensities_concat = np.stack(intensities_concat, axis = 0)

        np.save(dataset_path+"inputs/" + file_naming("intensities", count) + ".npy", intensities_concat)
        count +=1
    
    count = 0
    for i in range(len(indices_list[0])):
        vx_list = []
        vy_list = []
        for j in range(len(indices_list)):
            idx = indices_list[j][i]
            vx = np.load(vel_path + file_naming("vx", idx)+ ".npy")
            vy = np.load(vel_path + file_naming("vy", idx)+ ".npy")
            vx_list.append(vx)
            vy_list.append(vy)

        vx_out = np.mean(vx_list, axis=0)
        vy_out = np.mean(vy_list, axis=0)

        velocities_concat = np.stack([vx_out, vy_out], axis = 0)
        np.save(dataset_path + "labels/" + file_naming("velocities", count) + ".npy", velocities_concat)
        count+=1

    
def crop_image_center (original_dir, input_data_name, new_dim, save_dir):
    """
    Crop the center of the image to a new dimension.
    Args:
        original_dir: Directory where the original image is located
        input_data_name: Name of the input data file
        new_dim: New dimension for the cropped image (assumed to be square)
        save_dir: Directory where the cropped image will be saved
    """

    input_data = np.load(os.path.join(original_dir, input_data_name))

    if input_data.ndim == 3:
        _, h, w = input_data.shape

    elif input_data.ndim == 4:
        _,_,h,w = input_data.shape
    cropped_img = []

    y_im_center = int(h/2)
    x_im_center = int(w/2)

    if input_data.ndim == 3:
        cropped_img = input_data[:, (y_im_center - int(new_dim/2)): (y_im_center + int(new_dim/2)), (x_im_center - int(new_dim/2)): (x_im_center + int(new_dim/2))]

    elif input_data.ndim == 4:
        cropped_img = input_data[:, :, (y_im_center - int(new_dim/2)): (y_im_center + int(new_dim/2)), (x_im_center - int(new_dim/2)): (x_im_center + int(new_dim/2))]

    #np.save(save_dir + input_data_name.replace(".npy", "_cropped.npy"), cropped_img)
    np.save(os.path.join(save_dir, input_data_name), cropped_img)

def crop_image_every_nth (n, original_dir, data_idx, new_dim, save_dir, sample_random = False):
    """
    Crop the image into smaller patches of size new_dim x new_dim, take every n-th patch.
    Args:
        n: Step size for cropping (every n-th patch will be saved)
        original_dir: Directory where the original image is located
        data_idx: Index of the data file to be cropped
        new_dim: Dimension of the cropped patches (assumed to be square)
        save_dir: Directory where the cropped patches will be saved
        sample_random: If True, sample patches randomly instead of uniformly
    """

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
        for i in range(0, len(cropped_img_inputs), n):
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])
    
    else: #sample randomly
        to_save = int(len(cropped_img_inputs) / n)
        indices_to_save = random.sample(range(0, len(cropped_img_inputs)), to_save)

        for i in indices_to_save:
            np.save(save_dir + 'inputs/intensities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_inputs[i])
            np.save(save_dir + 'labels/velocities_' + data_idx + '_cropped_' + str (i) + ".npy", cropped_img_labels[i])

def normalize_layerwise (input_data):
    '''
    Normalizes each datapoint separately, layers all together

    Args:
        input_data: Input data to be normalized (2-channel tensor or numpy array)
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
    Args:
        output_path: Path where the original velocity files are located
        save_path: Path where the normalized velocity files will be saved
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
    '''
    Denormalizes data using the provided mean and standard deviation.
    Args:
        data: Normalized data to be denormalized (tensor or numpy array)
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    '''
    return data*std + mean

def get_doppler_width(temperature, central_wavelength, atomic_mass):
    '''
    Calculate the Doppler width for a given temperature, central wavelength, and atomic mass.

    Args:
        temperature: Temperature in Kelvin
        central_wavelength: Central wavelength in Angstroms
        atomic_mass: Atomic mass in atomic mass units (amu) 
    Returns:
        Doppler width in Angstroms
    '''
    k_B = 1.380649e-16  # Boltzmann constant in erg/K
    c = 2.99792458e10   # Speed of light in cm/s
    m_u = 1.66053906660e-24  # Atomic mass unit in grams
    atomic_mass_g = atomic_mass * m_u  # Convert atomic mass to grams
    doppler_width = central_wavelength * 1e-8 / c * np.sqrt(2 * k_B * temperature / atomic_mass_g)  # Doppler width in cm
    doppler_width_angstrom = doppler_width * 1e8  # Convert Doppler width to Angstroms
    return doppler_width_angstrom

def log_sampling_doppler_w(central_wavelengths, doppler_widths, n_samples = 12, m_min=0.05, m_max=3.0):

    one_side = n_samples // 2
    log_space = np.logspace(np.log10(m_min), np.log10(m_max), one_side)
    offsets = np.concatenate((-log_space[::-1], [0.0], log_space))
    logspacing = np.array([central_wavelengths[i] + doppler_widths[i] * offsets for i in range(len(central_wavelengths))])

    new_wave_grid = [logspacing[i] for i in range(len(logspacing))]
    
    return new_wave_grid

def create_dummy_datacubes(taus, save_path, dataset_version,murampath = '/dat/milic/2D', snapi_path = '/dat/milic/3D/3D_snapi_spectra/'):
    '''
    Creates full maps for dummy data:

    - Stokes I, V cubes go from 0 - 4500 iteration, every 150 iterations (i.e. cadence 30s)
    - Velocity will have the same cadence and go also (in general)from 0 - 4500 iteration
    - Since the velocities are calculated for the middle of the interval, if for the input it is taken 0 and 150 iteration, 
    then the velocity is calculated in the middle of 50 and 100 (because it is the closest to the middle of the input interval)
    '''

    print("Generating inputs...")

    #LOG SAMPLING IN INDICES

    h, w, c, l = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_0.fits'))[0].data.shape
    n_samples = [21,11,11] #for v1 and v2
    # n_samples = [42, 22, 22] #for v3
    # n_samples = [84, 44, 44] # for v4
    # n_samples = [252, 132, 132] #for v5
    line_centers = [269, 542, 586]
    line_widths = [250, 50, 50]
    # indices = [np.logspace(np.log10(1), np.log10(line_widths[i]//2), n_samples[i]//2) for i in range(len(n_samples))]
    # indices = [np.round(ind).astype(int) for ind in indices]
    # offsets = [np.concatenate((-ind[::-1], [0], ind)) for ind in indices]
    # individual_grids = [offsets[i] + line_centers[i] for i in range(len(line_centers))] 
    # new_index_grid = [np.clip(ig, 0, l-1) for ig in individual_grids]
    # np.save(os.path.join(save_path, dataset_version, 'downsampled_grids', 'indices_downsampled.npy'), np.concatenate(new_index_grid))
    
    input_data_i, input_data_v = [], []
    intensities = []
    for i in range(0, 4650, 150):
        #NOTE PROJECT CHECK VERSION:
        I = muram.MuramIntensity(murampath, i)[::2, ::2]
        intensities.append(I)

        #NOTE THESIS VERSION:
        # input_data = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_{i}.fits'))[0].data

        #sample logarithmically around the centers of the spectral lines
        # stokes_I = input_data[:, :, 0, ::4]
        # if input_data.shape[2] ==4:
        #     stokes_V = input_data[:, :, 3, ::4]
        # elif input_data.shape[2] == 2:
        #     stokes_V = input_data[:, :, 1, ::4]
        # # stokes_I_g, stokes_V_g = [], []

        # # for g in new_index_grid:
        # #     stokes_I_g.append(stokes_I[:, :, g])
        # #     stokes_V_g.append(stokes_V[:, :, g])
       
        # # stokes_I_d = np.concatenate(stokes_I_g, axis=2)
        # # stokes_V_d = np.concatenate(stokes_V_g, axis=2)

        # # I_t = np.transpose(stokes_I_d, (2, 0, 1))  
        # # V_t = np.transpose(stokes_V_d, (2, 0, 1)) 

    #     I_t = np.transpose(stokes_I, (2, 0, 1))  
    #     V_t = np.transpose(stokes_V, (2, 0, 1)) 
        
    #     input_data_i.append(I_t)
    #     input_data_v.append(V_t)

    # I_save = np.stack(input_data_i, axis=0)
    # V_save = np.stack(input_data_v, axis=0)
    # print(f"Input I shape after downsampling: {I_save.shape}")
    # print(f"Input V shape after downsampling: {V_save.shape}")
    # np.save(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_i_light_log.npy'), I_save)
    # np.save(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_v_light_log.npy'), V_save)
    np.save(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_intensities.npy'), np.stack(intensities, axis=0))

    # OUTPUTS
    print("Generating labels...")
    # v1_iters = np.arange(50, 4550, 150)
    # v2_iters = np.arange(100, 4600, 150) #normally

    #for v5, v6_1, v7, v8:
    # v1_iters = np.arange(0, 4650, 150)

    # for tau in taus:
    #     vel_data = []
    #     for i in range(len(v1_iters)):
    #         v1_idx = v1_iters[i]
            # v2_idx = v2_iters[i]

            # muram_v1 = mio.MuramTauSlice(murampath, v1_idx, tau)
            # muram_v2 = mio.MuramTauSlice(murampath, v2_idx, tau)

            # vx1 = muram_v1.vy[::2,::2]
            # vy1 = muram_v1.vz[::2,::2]
            # vz1 = muram_v1.vx[::2,::2]

            # vx2 = muram_v2.vy[::2,::2]
            # vy2 = muram_v2.vz[::2,::2]
            # vz2 = muram_v2.vx[::2,::2]

            # vx = (vx1 + vx2) / 2.
            # vy = (vy1 + vy2) / 2.
            # vz = (vz1 + vz2) / 2.

            #for v5, v6_1, v7, v8:
        #     vx = vx1
        #     vy = vy1
        #     vz = vz1

        #     v_out = np.stack([vx, vy, vz], axis=0)
        #     vel_data.append(v_out)
        # vel_data = np.stack(vel_data, axis=0)
        # print(f"Velocity data shape: {vel_data.shape}")
        # np.save(os.path.join(save_path, dataset_version, f'stacked/denorm/stacked_velocities_{tau}.npy'), vel_data)

    #NORMALIZE DATA AND SAVE STATS
    print("Normalizing data...")
    #NOTE PROJECT CHECK VERSION
    intensities_data = np.load(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_intensities.npy'))
    mean_intensities = np.mean(intensities_data)
    std_intensities = np.std(intensities_data)
    norm_intensities = (intensities_data - mean_intensities) / std_intensities
    np.save(os.path.join(save_path, dataset_version, 'stacked/norm/stacked_intensities_normalized.npy'), norm_intensities)

    #NOTE THESIS VERSION
    # input_data_i = np.load(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_i_light_log.npy'))
    # input_data_v = np.load(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_v_light_log.npy'))
    
    # mean_i = np.mean(input_data_i)
    # std_i = np.std(input_data_i)
    # norm_input_i = (input_data_i - mean_i) / std_i
    # np.save(os.path.join(save_path, dataset_version, 'stacked/norm/stacked_inputs_i_light_log_normalized.npy'), norm_input_i)

    # mean_v = np.mean(input_data_v)
    # std_v = np.std(input_data_v)
    # norm_input_v = (input_data_v - mean_v) / std_v
    # np.save(os.path.join(save_path, dataset_version, 'stacked/norm/stacked_inputs_v_light_log_normalized.npy'), norm_input_v)

    # for tau in taus:
    #     vel_data = np.load(os.path.join(save_path, dataset_version, f'stacked/denorm/stacked_velocities_{tau}.npy'))
    #     mean_vel = np.mean(vel_data)
    #     std_vel = np.std(vel_data)
    #     norm_vel = (vel_data - mean_vel) / std_vel
    #     np.save(os.path.join(save_path, dataset_version, f'stacked/norm/stacked_velocities_{tau}_normalized.npy'), norm_vel)
        
    #     with open(f'important_stats/normalization_stats_{dataset_version}.txt', 'a') as f:
    #         f.write(f"Velocities mean tau = {tau}: {mean_vel}, std: {std_vel}\n")

    with open(f'important_stats/normalization_stats_{dataset_version}.txt', 'a') as f:
        # f.write(f"Input Stokes I mean: {mean_i}, std: {std_i}\n")
        # f.write(f"Input Stokes V mean: {mean_v}, std: {std_v}\n")
        f.write(f"Intensities mean: {mean_intensities}, std: {std_intensities}\n")

def create_downsampled_data(num_samples_per_channel, num_spatial_samples, dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):
    cube_path = os.path.join(main_path, dataset_version, "stacked/norm/")
    save_path = os.path.join(main_path, dataset_version, "dataset/train/")

    # sample random coordinates
    data_ch = 2
    data_h_w = 64

    max_ch = int(31 - 2 - data_ch//2) #-2 to avoid having test data in the training set
    max_h_w = 768 - data_h_w//2
    min_ch = data_ch//2
    min_h_w = data_h_w//2

    # center_h = random.sample(range(min_h_w, max_h_w), num_spatial_samples)
    # center_w = random.sample(range(min_h_w, max_h_w), num_spatial_samples)
    # center_c = random.sample(range(min_ch, max_ch), num_samples_per_channel)
    
    #for v1:
    center_c=[11, 17, 6, 20, 18, 10, 7, 13, 1, 22, 21, 14, 12, 19, 26, 5, 9, 25, 16, 23]
    center_h=[360, 598, 349, 111, 675, 583, 640, 155, 702, 589, 82, 691, 358, 313, 590, 69, 564, 357, 253, 570, 181, 662, 175, 486, 713, 239, 114, 347, 573, 480, 317, 170, 342, 647, 601, 462, 714, 502, 356, 279, 343, 520, 157, 443, 103, 483, 182, 426, 723, 54, 203, 280, 361, 167, 39, 117]
    center_w =[339, 528, 285, 241, 568, 95, 46, 544, 714, 337, 534, 678, 696, 96, 546, 255, 600, 118, 129, 582, 591, 407, 268, 169, 113, 184, 552, 196, 674, 146, 387, 89, 426, 75, 473, 298, 104, 499, 597, 704, 428, 247, 148, 231, 471, 486, 429, 452, 44, 477, 423, 52, 186, 700, 585, 262]

    # save the rand coordinates for future reference
    # with open(f'important_stats/random_centers_{dataset_version}.txt', 'w') as f:
    #     f.write("center_c: " + ", ".join(map(str, center_c)) + "\n")
    #     f.write("center_h: " + ", ".join(map(str, center_h)) + "\n")
    #     f.write("center_w: " + ", ".join(map(str, center_w)) + "\n")

    # stokes_I = np.load(os.path.join(cube_path, 'stacked_inputs_i_light_log_normalized.npy'))
    # stokes_V = np.load(os.path.join(cube_path, 'stacked_inputs_v_light_log_normalized.npy'))
    # vel_1 = np.load(os.path.join(cube_path, 'stacked_velocities_1.0_normalized.npy'))
    # vel_1e_1 = np.load(os.path.join(cube_path, 'stacked_velocities_0.1_normalized.npy'))
    # vel_1e_2 = np.load(os.path.join(cube_path, 'stacked_velocities_0.01_normalized.npy'))
    # vel_1e_3 = np.load(os.path.join(cube_path, 'stacked_velocities_0.001_normalized.npy'))
    # vel_1e_4 = np.load(os.path.join(cube_path, 'stacked_velocities_0.0001_normalized.npy'))

    intensities = np.load(os.path.join(cube_path, 'stacked_intensities_normalized.npy'))
    
    half_s = data_h_w // 2
    half_c = data_ch // 2

    for i in range(num_spatial_samples):
        print("Processing center: ", i)
        for j in range(num_spatial_samples):
            for k in range(num_samples_per_channel):
                ch = center_h[i]
                cw = center_w[j]
                cc = center_c[k]

                intensity = intensities[cc-half_c:cc+half_c, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                np.save(save_path + f'inputs/intensities/intensities_{cc}_{ch}_{cw}.npy', intensity)

                # stokesI = stokes_I[cc-half_c:cc+half_c, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # stokesV = stokes_V[cc-half_c:cc+half_c, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s] #normally
                # vel1 = vel_1[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_1 = vel_1e_1[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_2 = vel_1e_2[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_3 = vel_1e_3[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_4 = vel_1e_4[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]

                #for v5, v6_1, v7, v8:
                # stokesI = stokes_I[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # stokesV = stokes_V[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                
                # vel1 = vel_1[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_1 = vel_1e_1[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_2 = vel_1e_2[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_3 = vel_1e_3[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                # vel1e_4 = vel_1e_4[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]

                # np.save(save_path + f'inputs/stokes_I/stokes_I_{cc}_{ch}_{cw}.npy', stokesI)
                # np.save(save_path + f'inputs/stokes_V/stokes_V_{cc}_{ch}_{cw}.npy', stokesV)
                # np.save(save_path + f'labels/tau_1.0/velocities_{cc}_{ch}_{cw}.npy', vel1)
                # np.save(save_path + f'labels/tau_1e-1/velocities_{cc}_{ch}_{cw}.npy', vel1e_1)
                # np.save(save_path + f'labels/tau_1e-2/velocities_{cc}_{ch}_{cw}.npy', vel1e_2)
                # np.save(save_path + f'labels/tau_1e-3/velocities_{cc}_{ch}_{cw}.npy', vel1e_3)
                # np.save(save_path + f'labels/tau_1e-4/velocities_{cc}_{ch}_{cw}.npy', vel1e_4)

def create_last_2_layers_test_data (dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):
    cube_norm_path = os.path.join(main_path, dataset_version, "stacked/norm/")
    save_path = os.path.join(main_path, dataset_version, "dataset/test/last_2_layers/")
    intensities = np.load(os.path.join(cube_norm_path, 'stacked_intensities_normalized.npy'))
    # stokes_I = np.load(os.path.join(cube_norm_path, 'stacked_inputs_i_light_log_normalized.npy'))
    # stokes_V = np.load(os.path.join(cube_norm_path, 'stacked_inputs_v_light_log_normalized.npy'))
    
    # vel_1 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_1_normalized.npy'))
    # vel_1e_1 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.1_normalized.npy'))
    # vel_1e_2 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.01_normalized.npy'))
    # vel_1e_3 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.001_normalized.npy'))
    # vel_1e_4 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.0001_normalized.npy'))

    intensities_last_2 = intensities[-2:, :, :]
    # stokes_I_last2 = stokes_I[-2:, :, :, :]
    # stokes_V_last2 = stokes_V[-2:, :, :, :]
    # vel_1_last2 = vel_1[-1:, :2, :, :]
    # vel_1e_1_last2 = vel_1e_1[-1:, :, :, :]
    # vel_1e_2_last2 = vel_1e_2[-1:, :, :, :]
    # vel_1e_3_last2 = vel_1e_3[-1:, :, :, :]
    # vel_1e_4_last2 = vel_1e_4[-1:, :, :, :]

    #NOTE only v5, v7, v6_1, v8 version
    # stokes_I_last2 = stokes_I[-1, :, :, :]
    # stokes_V_last2 = stokes_V[-1, :, :, :]

    np.save(os.path.join(save_path, 'inputs/intensities/intensities_2.npy'), intensities_last_2)
    # np.save(os.path.join(save_path, 'inputs/stokes_I/stokes_I_2.npy'), stokes_I_last2)
    # np.save(os.path.join(save_path, 'inputs/stokes_V/stokes_V_2.npy'), stokes_V_last2)
    # np.save(os.path.join(save_path, 'labels/tau_1.0/velocities_tau_1.0.npy'), vel_1_last2)
    # np.save(os.path.join(save_path, 'labels/tau_1e-1/velocities_tau_1e-1.npy'), vel_1e_1_last2)
    # np.save(os.path.join(save_path, 'labels/tau_1e-2/velocities_tau_1e-2.npy'), vel_1e_2_last2)
    # np.save(os.path.join(save_path, 'labels/tau_1e-3/velocities_tau_1e-3.npy'), vel_1e_3_last2)
    # np.save(os.path.join(save_path, 'labels/tau_1e-4/velocities_tau_1e-4.npy'), vel_1e_4_last2)

def crop_center_4_patches(original_dir, input_data_name, new_dim, save_dir):
    '''
    Crop the center of the image into 4 patches of size new_dim x new_dim.
    Args:
        original_dir: Directory where the original image is located
        input_data_name: Name of the input data file
        new_dim: Dimension for the cropped patches (assumed to be square)
        save_dir: Directory where the cropped patches will be saved
    '''

    input_data = np.load(os.path.join(original_dir, input_data_name))

    if input_data.ndim == 3:
        _, h, w = input_data.shape

    elif input_data.ndim == 4:
        _,_,h,w = input_data.shape

    y_im_center = int(h/2)
    x_im_center = int(w/2)

    if input_data.ndim == 3:
        cropped_img1 = input_data[:, (y_im_center - new_dim): y_im_center, (x_im_center - new_dim): x_im_center]
        cropped_img2 = input_data[:, (y_im_center - new_dim): y_im_center, x_im_center: (x_im_center + new_dim)]
        cropped_img3 = input_data[:, y_im_center: (y_im_center + new_dim), (x_im_center - new_dim): x_im_center]
        cropped_img4 = input_data[:, y_im_center: (y_im_center + new_dim), x_im_center: (x_im_center + new_dim)]

    elif input_data.ndim == 4:
        cropped_img1 = input_data[:, :, (y_im_center - new_dim): y_im_center, (x_im_center - new_dim): x_im_center]
        cropped_img2 = input_data[:, :, (y_im_center - new_dim): y_im_center, x_im_center: (x_im_center + new_dim)]
        cropped_img3 = input_data[:, :, y_im_center: (y_im_center + new_dim), (x_im_center - new_dim): x_im_center]
        cropped_img4 = input_data[:, :, y_im_center: (y_im_center + new_dim), x_im_center: (x_im_center + new_dim)]

    np.save(os.path.join(save_dir, input_data_name.replace(".npy", "_1.npy")), cropped_img1)
    np.save(os.path.join(save_dir, input_data_name.replace(".npy", "_2.npy")), cropped_img2)
    np.save(os.path.join(save_dir, input_data_name.replace(".npy", "_3.npy")), cropped_img3)
    np.save(os.path.join(save_dir, input_data_name.replace(".npy", "_4.npy")), cropped_img4)

def create_dataset_v2(v1_path = "/dat/xenoss/thesis/data/v1/dataset/train", save_path = "/dat/xenoss/thesis/data/v2/dataset/train/"):
    '''
    v2 dataset consists of the data sampled in the same way as v1, but with the resolution 16x16.
    The data is made from cropping the central 16x16 patches from v1 stokes I, stokes V and velocities.
    '''
    stokes_I_path = os.path.join(v1_path, 'inputs/stokes_I/')
    stokes_V_path = os.path.join(v1_path, 'inputs/stokes_V/')
    velocities_path = os.path.join(v1_path, 'labels/')

    taus = ["tau_1e-1"] #for now I only test on tau=1e-1

    stokes_I = os.listdir(stokes_I_path)
    stokes_V = os.listdir(stokes_V_path)

    print("Creating Stokes I cropped data...")
    for si in stokes_I:
        crop_center_4_patches(stokes_I_path, si, 16, os.path.join(save_path, 'inputs/stokes_I/'))
    print("Creating Stokes V cropped data...")
    for sv in stokes_V:
        crop_center_4_patches(stokes_V_path, sv, 16, os.path.join(save_path, 'inputs/stokes_V/'))
    print("Creating velocities cropped data...")
    for t in taus:
        print("Processing tau: ", t)
        velocities_specific_tau_path = os.path.join(velocities_path, t)
        velocities_specific_tau = os.listdir(velocities_specific_tau_path)
        for v in velocities_specific_tau:
            crop_center_4_patches(velocities_specific_tau_path, v, 16, os.path.join(save_path, 'labels/', t))

def make_dataset_compatible_with_project_version (dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):

    dataset_path = os.path.join(main_path, dataset_version, "dataset/train/")
    save_path = os.path.join(main_path, dataset_version+"_transformed", "dataset/train/")

    stokes_I_path = os.path.join(dataset_path, 'inputs/stokes_I/')
    stokes_V_path = os.path.join(dataset_path, 'inputs/stokes_V/')
    velocities_path = os.path.join(dataset_path, 'labels/')

    taus = ["1.0"]

    # stokes_I_files = os.listdir(stokes_I_path)
    # stokes_V_files = os.listdir(stokes_V_path)

    # print("Transforming Stokes I files...")
    # for si in stokes_I_files:
    #     I = np.load(os.path.join(stokes_I_path, si))[:,0,:,:]
    #     np.save(os.path.join(save_path, 'inputs/stokes_I/',si), I)

    # print("Transforming Stokes V files...")
    # for sv in stokes_V_files:
    #     V = np.load(os.path.join(stokes_V_path, sv))[:,0,:,:]
    #     np.save(os.path.join(save_path, 'inputs/stokes_V/', sv), V)

    print("Transforming velocities files...")
    for t in taus:
        print("Processing tau: ", t)
        velocities_tau_path = os.path.join(velocities_path, f"tau_{t}")
        velocities_tau_files = os.listdir(velocities_tau_path)

        for v in velocities_tau_files:
            vel = np.load(os.path.join(velocities_tau_path, v))[:2,:,:]
            np.save(os.path.join(save_path, 'labels/', f"tau_{t}/", v), vel)

def create_multiheight_dataset(num_samples_per_channel, num_spatial_samples, dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):
    '''
    For the first version of the multi-height dataset, I create a dataset with the same sampling as v1, 
    but with the 4 tau layers as labels: log tau = {0, -1, -2, -3} or 
    with 5 tau layers as labels: log tau = {0, -1, -2, -3, -4} (depending on the version).
    The input data is the same as in v1
    '''
    cube_path = os.path.join(main_path, dataset_version, "stacked/norm/")
    save_path = os.path.join(main_path, dataset_version, "dataset/train/")

    data_ch = 2
    data_h_w = 64

    max_ch = int(31 - 2 - data_ch//2) #-2 to avoid having test data in the training set
    max_h_w = 768 - data_h_w//2
    min_ch = data_ch//2
    min_h_w = data_h_w//2
    center_h = random.sample(range(min_h_w, max_h_w), num_spatial_samples)
    center_w = random.sample(range(min_h_w, max_h_w), num_spatial_samples)
    center_c = random.sample(range(min_ch, max_ch), num_samples_per_channel)

    #save the rand coordinates for future reference
    with open(f'important_stats/random_centers_{dataset_version}.txt', 'w') as f:
        f.write("center_c: " + ", ".join(map(str, center_c)) + "\n")
        f.write("center_h: " + ", ".join(map(str, center_h)) + "\n")
        f.write("center_w: " + ", ".join(map(str, center_w)) + "\n")

    stokes_I = np.load(os.path.join(cube_path, 'stacked_inputs_i_light_log_normalized.npy'))
    stokes_V = np.load(os.path.join(cube_path, 'stacked_inputs_v_light_log_normalized.npy'))
    
    vel_1 = np.load(os.path.join(main_path, dataset_version, 'stacked/norm/stacked_velocities_1_normalized.npy'))
    vel_1e_1 = np.load(os.path.join(main_path, dataset_version, 'stacked/norm/stacked_velocities_0.1_normalized.npy'))
    vel_1e_2 = np.load(os.path.join(main_path, dataset_version, 'stacked/norm/stacked_velocities_0.01_normalized.npy'))
    vel_1e_3 = np.load(os.path.join(main_path, dataset_version, 'stacked/norm/stacked_velocities_0.001_normalized.npy'))
    vel_1e_4 = np.load(os.path.join(main_path, dataset_version, 'stacked/norm/stacked_velocities_0.0001_normalized.npy'))

    half_s = data_h_w // 2
    half_c = data_ch // 2

    for i in range(num_spatial_samples):
        print("Processing center: ", i)
        for j in range(num_spatial_samples):
            for k in range(num_samples_per_channel):
                ch = center_h[i]
                cw = center_w[j]
                cc = center_c[k]

                stokesI = stokes_I[cc-half_c:cc+half_c, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                stokesV = stokes_V[cc-half_c:cc+half_c, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s] #normally
                vel1 = vel_1[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_1 = vel_1e_1[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_2 = vel_1e_2[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_3 = vel_1e_3[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_4 = vel_1e_4[cc-1, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]

                vel = np.stack([vel1, vel1e_1, vel1e_2, vel1e_3, vel1e_4], axis=0)

                np.save(save_path + f'inputs/stokes_I/stokes_I_{cc}_{ch}_{cw}.npy', stokesI)
                np.save(save_path + f'inputs/stokes_V/stokes_V_{cc}_{ch}_{cw}.npy', stokesV)
                np.save(save_path + f'labels/velocities_{cc}_{ch}_{cw}.npy', vel) 
            
if (__name__ == '__main__'):
    
    main_root = "/dat/xenoss/"
    taus = [1.0, 0.1, 0.01, 0.001, 0.0001]
    # create_dummy_datacubes(taus=taus, save_path=main_root + "thesis/data/", dataset_version="v1")
    # create_downsampled_data(num_samples_per_channel=20, num_spatial_samples=56, dataset_version="v1")
    create_last_2_layers_test_data(dataset_version="v1")

    # create_multiheight_dataset(num_samples_per_channel=12, num_spatial_samples=56, dataset_version="v12")

    