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
import astropy.io.fits as fits
import scipy.interpolate as sci

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

def create_dummy_datacubes(tau, save_path, dataset_version,murampath = '/dat/milic/2D', snapi_path = '/dat/milic/3D/3D_snapi_spectra/'):
    '''
    Creates full maps for dummy data:

    - Stokes I, V cubes go from 0 - 4500 iteration, every 150 iterations (i.e. cadence 30s)
    - Velocity will have the same cadence and go also (in general)from 0 - 4500 iteration
    - Since the velocities are calculated for the middle of the interval, if for the input it is taken 0 and 150 iteration, 
    then the velocity is calculated in the middle of 50 and 100 (because it is the closest to the middle of the input interval)
    '''

    # INPUTS - NO DOWNSAMPLING
    print("Generating inputs...")
    # input_indices1 = np.arange(0,4500,150)
    # input_indices2 = np.arange(150,4650,150)
    # input_data_i, input_data_v = [], []
    # for i in range(len(input_indices1)):
    #     input1 = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_{input_indices1[i]}.fits'))[0].data
    #     input2 = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_{input_indices2[i]}.fits'))[0].data

    #     stokes_I1 = input1[:, :, 0, :]
    #     stokes_V1 = input1[:, :, 1, :]
    #     stokes_I2 = input2[:, :, 0, :]
    #     stokes_V2 = input2[:, :, 1, :]

    #     I1_t = np.transpose(stokes_I1, (2, 0, 1))  
    #     I2_t = np.transpose(stokes_I2, (2, 0, 1))  
    #     input_I = np.stack([I1_t, I2_t], axis=0)

    #     V1_t = np.transpose(stokes_V1, (2, 0, 1))
    #     V2_t = np.transpose(stokes_V2, (2, 0, 1))
    #     input_V = np.stack([V1_t, V2_t], axis=0)

    #     np.save(os.path.join(save_path, 'inputs/stokes_I', f'stokesI_{input_indices1[i]}_{input_indices2[i]}'), np.array(input_I))
    #     np.save(os.path.join(save_path, 'inputs/stokes_V', f'stokesV_{input_indices1[i]}_{input_indices2[i]}'), np.array(input_V))

    #DOWNSAMPLE INPUT CUBES
    # pick n points and do log sampling with the n points on one side and symmetrical on the other. then round the logs up so i dont have to interpolate
    # print("Downsampling inputs...")

    #LOG SAMPLING IN INDICES

    h, w, c, l = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_0.fits'))[0].data.shape
    # n_samples = [21,11,11] #for v1 and v2
    # n_samples = [42, 22, 22] #for v3
    n_samples = [84, 44, 44] # for v4
    line_centers = [269, 542, 586]
    line_widths = [250, 50, 50]
    indices = [np.logspace(np.log10(1), np.log10(line_widths[i]//2), n_samples[i]//2) for i in range(len(n_samples))]
    indices = [np.round(ind).astype(int) for ind in indices]
    offsets = [np.concatenate((-ind[::-1], [0], ind)) for ind in indices]
    individual_grids = [offsets[i] + line_centers[i] for i in range(len(line_centers))] 
    new_index_grid = [np.clip(ig, 0, l-1) for ig in individual_grids]
    np.save(os.path.join(save_path, dataset_version, 'downsampled_grids', 'indices_downsampled.npy'), np.concatenate(new_index_grid))
    
    input_data_i, input_data_v = [], []
    for i in range(0, 4650, 150):
        input_data = fits.open(os.path.join(snapi_path, f'tumag_stokesIV_cube_{i}.fits'))[0].data

        #sample logarithmically around the centers of the spectral lines
        stokes_I = input_data[:, :, 0, :]
        stokes_V = input_data[:, :, 1, :]
        stokes_I_g, stokes_V_g = [], []

        for g in new_index_grid:
            stokes_I_g.append(stokes_I[:, :, g])
            stokes_V_g.append(stokes_V[:, :, g])
       
        stokes_I_d = np.concatenate(stokes_I_g, axis=2)
        stokes_V_d = np.concatenate(stokes_V_g, axis=2)

        I_t = np.transpose(stokes_I_d, (2, 0, 1))  
        V_t = np.transpose(stokes_V_d, (2, 0, 1)) 
        # np.save(os.path.join(save_path, 'inputs/downsampled_cubes/stokes_I', f'stokesI_{i}_downsampled'), I_t)
        # np.save(os.path.join(save_path, 'inputs/downsampled_cubes/stokes_V', f'stokesV_{i}_downsampled'), V_t)
        
        input_data_i.append(I_t)
        input_data_v.append(V_t)

    I_save = np.stack(input_data_i, axis=0)
    V_save = np.stack(input_data_v, axis=0)
    print(f"Input I shape after downsampling: {I_save.shape}")
    print(f"Input V shape after downsampling: {V_save.shape}")
    np.save(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_i_light_log.npy'), I_save)
    np.save(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_v_light_log.npy'), V_save)

    # OUTPUTS
    # print("Generating labels...")
    # v1_iters = np.arange(50, 4550, 150)
    # v2_iters = np.arange(100, 4600, 150)

    # vel_data = []
    # for i in range(len(v1_iters)):
    #     v1_idx = v1_iters[i]
    #     v2_idx = v2_iters[i]

    #     muram_v1 = mio.MuramTauSlice(murampath, v1_idx, tau)
    #     muram_v2 = mio.MuramTauSlice(murampath, v2_idx, tau)

    #     vx1 = muram_v1.vy[::2,::2]
    #     vy1 = muram_v1.vz[::2,::2]
    #     vz1 = muram_v1.vx[::2,::2]

    #     vx2 = muram_v2.vy[::2,::2]
    #     vy2 = muram_v2.vz[::2,::2]
    #     vz2 = muram_v2.vx[::2,::2]

    #     vx = (vx1 + vx2) / 2.
    #     vy = (vy1 + vy2) / 2.
    #     vz = (vz1 + vz2) / 2.

    #     v_out = np.stack([vx, vy, vz], axis=0)
    #     #np.save(os.path.join(save_path, 'labels', f'vel_{v1_idx}_{v2_idx}.npy'), v_out)
    #     vel_data.append(v_out)
    # vel_data = np.stack(vel_data, axis=0)
    # print(f"Velocity data shape: {vel_data.shape}")
    # np.save(os.path.join(save_path, dataset_version, f'stacked/denorm/stacked_velocities_{tau}.npy'), vel_data)

    #NORMALIZE DATA AND SAVE STATS
    print("Normalizing data...")
    input_data_i = np.load(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_i_light_log.npy'))
    input_data_v = np.load(os.path.join(save_path, dataset_version, 'stacked/denorm/stacked_inputs_v_light_log.npy'))
    vel_data = np.load(os.path.join(save_path, dataset_version, f'stacked/denorm/stacked_velocities_{tau}.npy'))

    mean_i = np.mean(input_data_i)
    std_i = np.std(input_data_i)
    norm_input_i = (input_data_i - mean_i) / std_i
    np.save(os.path.join(save_path, dataset_version, 'stacked/norm/stacked_inputs_i_light_log_normalized.npy'), norm_input_i)

    mean_v = np.mean(input_data_v)
    std_v = np.std(input_data_v)
    norm_input_v = (input_data_v - mean_v) / std_v
    np.save(os.path.join(save_path, dataset_version, 'stacked/norm/stacked_inputs_v_light_log_normalized.npy'), norm_input_v)

    # mean_vel = np.mean(vel_data)
    # std_vel = np.std(vel_data)
    # norm_vel = (vel_data - mean_vel) / std_vel
    # np.save(os.path.join(save_path, dataset_version, f'stacked/norm/stacked_velocities_{tau}_normalized.npy'), norm_vel)

    with open(f'important_stats/normalization_stats_{dataset_version}.txt', 'a') as f:
        f.write(f"Input Stokes I mean (log sampled): {mean_i}, std: {std_i}\n")
        f.write(f"Input Stokes V mean (log sampled): {mean_v}, std: {std_v}\n")
        # f.write(f"Velocities mean tau = {tau}: {mean_vel}, std: {std_vel}\n")

def create_downsampled_data(num_samples_per_channel, num_spatial_samples, dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):
    cube_path = os.path.join(main_path, dataset_version, "stacked/norm/")
    save_path = os.path.join(main_path, dataset_version, "dataset/train/")
    # sample random coordinates
    data_ch = 2
    #data_h_w = 64
    data_h_w = 16

    max_ch = 31 - data_ch//2
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
    #velocities = np.load(os.path.join(cube_path, 'stacked_velocities_normalized.npy'))
    vel_1 = np.load(os.path.join(cube_path, 'stacked_velocities_1_normalized.npy'))
    vel_1e_1 = np.load(os.path.join(cube_path, 'stacked_velocities_0.1_normalized.npy'))
    vel_1e_2 = np.load(os.path.join(cube_path, 'stacked_velocities_0.01_normalized.npy'))
    vel_1e_3 = np.load(os.path.join(cube_path, 'stacked_velocities_0.001_normalized.npy'))
    vel_1e_4 = np.load(os.path.join(cube_path, 'stacked_velocities_0.0001_normalized.npy'))

    # center_c = [8, 10, 23, 28, 15, 18, 27, 5, 7, 20, 22, 14, 29, 21, 12, 2, 13, 26, 11, 3, 19, 4]
    # center_h = [172, 454, 135, 654, 156, 96, 682, 142, 295, 387, 160, 473, 76, 87, 557, 685, 430, 90, 604, 639, 699, 355]
    # center_w = [525, 681, 687, 519, 698, 408, 228, 261, 655, 223, 468, 220, 680, 92, 669, 538, 134, 601, 204, 635, 474, 152]
    
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
                stokesV = stokes_V[cc-half_c:cc+half_c, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                #velocity = velocities[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                
                vel1 = vel_1[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_1 = vel_1e_1[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_2 = vel_1e_2[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_3 = vel_1e_3[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]
                vel1e_4 = vel_1e_4[cc, :, ch - half_s: ch + half_s, cw - half_s: cw + half_s]

                np.save(save_path + f'inputs/stokes_I/stokes_I_{cc}_{ch}_{cw}.npy', stokesI)
                np.save(save_path + f'inputs/stokes_V/stokes_V_{cc}_{ch}_{cw}.npy', stokesV)
                np.save(save_path + f'labels/tau_1.0/velocities_{cc}_{ch}_{cw}.npy', vel1)
                np.save(save_path + f'labels/tau_1e-1/velocities_{cc}_{ch}_{cw}.npy', vel1e_1)
                np.save(save_path + f'labels/tau_1e-2/velocities_{cc}_{ch}_{cw}.npy', vel1e_2)
                np.save(save_path + f'labels/tau_1e-3/velocities_{cc}_{ch}_{cw}.npy', vel1e_3)
                np.save(save_path + f'labels/tau_1e-4/velocities_{cc}_{ch}_{cw}.npy', vel1e_4)
                #np.save(save_path + f'labels/velocities_{cc}_{ch}_{cw}.npy', velocity)

def create_last_2_layers_test_data (dataset_version, main_path = "/home/xenoss/dat/thesis/data/"):
    cube_norm_path = os.path.join(main_path, dataset_version, "stacked/norm/")
    save_path = os.path.join(main_path, dataset_version, "dataset/test/last_2_layers/")
    stokes_I = np.load(os.path.join(cube_norm_path, 'stacked_inputs_i_light_log_normalized.npy'))
    stokes_V = np.load(os.path.join(cube_norm_path, 'stacked_inputs_v_light_log_normalized.npy'))
    
    vel_1 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_1_normalized.npy'))
    vel_1e_1 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.1_normalized.npy'))
    vel_1e_2 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.01_normalized.npy'))
    vel_1e_3 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.001_normalized.npy'))
    vel_1e_4 = np.load(os.path.join(cube_norm_path, 'stacked_velocities_0.0001_normalized.npy'))

    stokes_I_last2 = stokes_I[-2:, :, :, :]
    stokes_V_last2 = stokes_V[-2:, :, :, :]
    vel_1_last2 = vel_1[-1:, :, :, :]
    vel_1e_1_last2 = vel_1e_1[-1:, :, :, :]
    vel_1e_2_last2 = vel_1e_2[-1:, :, :, :]
    vel_1e_3_last2 = vel_1e_3[-1:, :, :, :]
    vel_1e_4_last2 = vel_1e_4[-1:, :, :, :]

    np.save(os.path.join(save_path, 'inputs/stokes_I/stokes_I_last2_layers.npy'), stokes_I_last2)
    np.save(os.path.join(save_path, 'inputs/stokes_V/stokes_V_last2_layers.npy'), stokes_V_last2)
    np.save(os.path.join(save_path, 'labels/tau_1.0/velocities_last2_layers.npy'), vel_1_last2)
    np.save(os.path.join(save_path, 'labels/tau_1e-1/velocities_last2_layers.npy'), vel_1e_1_last2)
    np.save(os.path.join(save_path, 'labels/tau_1e-2/velocities_last2_layers.npy'), vel_1e_2_last2)
    np.save(os.path.join(save_path, 'labels/tau_1e-3/velocities_last2_layers.npy'), vel_1e_3_last2)
    np.save(os.path.join(save_path, 'labels/tau_1e-4/velocities_last2_layers.npy'), vel_1e_4_last2)

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

if (__name__ == '__main__'):
    
    main_root = "/dat/xenoss/"
    #tau = [0.1, 0.01, 0.001, 0.0001]
    #for t in tau:
    #    create_dummy_datacubes(tau=t, save_path=main_root + "thesis/data/", dataset_version="v1")
    
    #create_downsampled_data(num_samples=25, dataset_version="v1")
    #create_last_2_layers_test_data(dataset_version="v1")
    #create_dataset_v2(v1_path = main_root + "thesis/data/v1/dataset/train", save_path = main_root + "thesis/data/v2/dataset/train/")
    #print(len(os.listdir(os.path.join(main_root, "thesis/data/v2/dataset/train/labels/tau_1e-1/"))))
    #data = os.listdir(os.path.join(main_root, "thesis/data/v3/dataset/train/inputs/stokes_V/"))
    #print(len(data))
    #print(np.load(os.path.join(main_root, "thesis/data/v2/dataset/train/inputs/stokes_V/", data[0])).shape)

    #FOR DATASET VERSION V3:
    #create_dummy_datacubes(tau=1e-1, save_path=main_root + "thesis/data/", dataset_version="v3")
    #create_downsampled_data(num_samples_per_channel=20, num_spatial_samples=56, dataset_version="v3")

    #FOR DATASET VERSION V4:
    #create_dummy_datacubes(tau=1e-1, save_path=main_root + "thesis/data/", dataset_version="v4")
    #create_downsampled_data(num_samples_per_channel=20, num_spatial_samples=56, dataset_version="v4")
    create_last_2_layers_test_data(dataset_version="v3")
    create_last_2_layers_test_data(dataset_version="v4")