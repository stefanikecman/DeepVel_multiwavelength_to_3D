import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy import stats
from scipy.stats import linregress

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_divergence(vx, vy, dx=0.016 * 1e6, dy=0.016 * 1e6):
    """
    Calculate the divergence of a 2D vector field (vx, vy).
    Args:
        vx: x-component of the vector field (2D tensor)
        vy: y-component of the vector field (2D tensor)
        dx: spacing in the x-direction (default in Mm)
        dy: spacing in the y-direction (default in Mm)
    Returns:
        Divergence of the vector field (2D tensor)
    """
    if isinstance(vx, np.ndarray):
        vx = torch.from_numpy(vx)
    if isinstance(vy, np.ndarray):
        vy = torch.from_numpy(vy)

    dvx_dx = torch.gradient(vx, spacing=dx, dim=1)[0]  # x derivative
    dvy_dy = torch.gradient(vy, spacing=dy, dim=0)[0]  # y derivative

    return dvx_dx + dvy_dy

def get_vorticity(vx, vy, dx=0.016 * 1e6, dy=0.016 * 1e6):
    """
    Calculate the vorticity of a 2D vector field (vx, vy).
    Args:
        vx: x-component of the vector field (2D tensor)
        vy: y-component of the vector field (2D tensor)
        dx: spacing in the x-direction (default in Mm)
        dy: spacing in the y-direction (default in Mm)
    Returns:
        Vorticity of the vector field (2D tensor)
    """

    
    if isinstance(vx, np.ndarray):
        vx = torch.from_numpy(vx)
    if isinstance(vy, np.ndarray):
        vy = torch.from_numpy(vy)

    dvy_dx = torch.gradient(vy, spacing=dx, dim=1)[0]  # x derivative of vy
    dvx_dy = torch.gradient(vx, spacing=dy, dim=0)[0]  # y derivative of vx

    return dvy_dx - dvx_dy

def denormalize_data(data, mean, std):
    """
    Denormalize the data using the provided mean and standard deviation.
    Args:
        data: Normalized data tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    Returns:
        Denormalized data tensor
    """
    return data * std + mean

def calculate_correlation (original, predicted):

    '''
    Pearson correlation coefficient - statistic:

    Ranges from -1 to 1.
    +1: perfect positive linear correlation
    0: no linear correlation
    -1: perfect negative linear correlation

    pvalue: 
    A p-value close to 0 means the observed correlation is highly statistically significant.
    '''
    original = original.cpu().numpy()
    predicted = predicted.cpu().numpy()

    pearson_coeffs = []
    print("Original shape:", original.shape)
    print("Predicted shape:", predicted.shape)

    # if predicted.shape[0] == 1: #infering only vz
    #     if isinstance(original, np.ndarray):
    #         original = torch.from_numpy(original)
    #     original = original[2, :, :].unsqueeze(0)
    #     original = original.cpu().numpy()
    n_channels = original.shape[0]

    for ch in range(n_channels):
        pearson_coeffs.append(stats.pearsonr(original[ch, :, :].flatten(), predicted[ch, :, :].flatten()))

    return pearson_coeffs

def get_all_metrics(original, predicted, is_divergence=False):
    """
    Calculate metrics for either velocity fields (3 channels) or divergence/vorticity (1 channel)
    Args:
        original: Original data tensor
        predicted: Predicted data tensor
        is_divergence: Boolean indicating if calculating metrics for divergence/vorticity
    """
    mse_l = nn.functional.mse_loss(predicted.to(device), original.to(device))
    rmse_l = torch.sqrt(mse_l)

    if is_divergence:
        orig_0 = original.squeeze().cpu().numpy()
        pred_0 = predicted.squeeze().cpu().numpy()

        slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(orig_0.flatten(), pred_0.flatten())
        pearson0 = stats.pearsonr(orig_0.flatten(), pred_0.flatten())[0]
        
        return {
            "mse": mse_l.item(), 
            "rmse": rmse_l.item(),
            "pearson": pearson0,
            "slope": slope_x
        }
    else:
        
        if original.shape[0] == 1 or predicted.shape[0] == 1:
            #this is the case for vz only model where we only have 1 channel in gt and pred
            orig_2 = original[0, :, :].cpu().numpy()
            pred_2 = predicted[0, :, :].cpu().numpy()
            slope_z, intercept_z, r_value_z, p_value_z, std_err_z = linregress(orig_2.flatten(), pred_2.flatten())
            pearsons = calculate_correlation(original, predicted)

            return {
                "mse": mse_l.item(), 
                "rmse": rmse_l.item(),
                "pearson_vz": pearsons[0].statistic,
                "slope_vz": slope_z
            }
        
        elif original.shape[0] == 2 or predicted.shape[0] == 2:
            orig_0 = original[0, :, :].cpu().numpy()
            orig_1 = original[1, :, :].cpu().numpy()
            
            pred_0 = predicted[0, :, :].cpu().numpy()
            pred_1 = predicted[1, :, :].cpu().numpy()
            

            slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(orig_0.flatten(), pred_0.flatten())
            slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(orig_1.flatten(), pred_1.flatten())
            

            pearsons = calculate_correlation(original, predicted)
            pearson0 = pearsons[0].statistic
            pearson1 = pearsons[1].statistic
            

            return {
                "mse": mse_l.item(), 
                "rmse": rmse_l.item(),
                "pearson_vx": pearson0.item(),
                "pearson_vy": pearson1.item(),
                "slope_vx": slope_x,
                "slope_vy": slope_y
            }

        else:
            orig_0 = original[0, :, :].cpu().numpy()
            orig_1 = original[1, :, :].cpu().numpy()
            orig_2 = original[2, :, :].cpu().numpy()
            pred_0 = predicted[0, :, :].cpu().numpy()
            pred_1 = predicted[1, :, :].cpu().numpy()
            pred_2 = predicted[2, :, :].cpu().numpy()

            slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(orig_0.flatten(), pred_0.flatten())
            slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(orig_1.flatten(), pred_1.flatten())
            slope_z, intercept_z, r_value_z, p_value_z, std_err_z = linregress(orig_2.flatten(), pred_2.flatten())

            pearsons = calculate_correlation(original, predicted)
            pearson0 = pearsons[0].statistic
            pearson1 = pearsons[1].statistic
            pearson2 = pearsons[2].statistic if len(pearsons) > 2 else None

            return {
                "mse": mse_l.item(), 
                "rmse": rmse_l.item(),
                "pearson_vx": pearson0.item(),
                "pearson_vy": pearson1.item(),
                "pearson_vz": pearson2.item(),
                "slope_vx": slope_x,
                "slope_vy": slope_y,
                "slope_vz": slope_z
            }