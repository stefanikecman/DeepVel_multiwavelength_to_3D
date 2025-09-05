import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy import stats
from scipy.stats import linregress

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_divergence(vx, vy, dx = 0.016 * 1e6, dy = 0.016 * 1e6):

    if isinstance(vx, torch.Tensor):
        vx = vx.detach().cpu().numpy()

    if isinstance(vy, torch.Tensor):
        vy = vy.detach().cpu().numpy()

    dvx_dx = np.gradient(vx, dx, axis=1)
    dvy_dy = np.gradient(vy, dy, axis=0)

    return torch.from_numpy(dvx_dx + dvy_dy)

def get_vorticity(vx, vy, dx = 0.016 * 1e6, dy = 0.016 * 1e6):

    if isinstance(vx, torch.Tensor):
        vx = vx.detach().cpu().numpy()

    if isinstance(vy, torch.Tensor):
        vy = vy.detach().cpu().numpy()

    dvy_dx = np.gradient(vy, dx, axis=1)
    dvx_dy = np.gradient(vx, dy, axis=0)

    return torch.from_numpy(dvy_dx - dvx_dy)

def div_vor_loss(pred, gt, alpha1=1.0, alpha2=72227.87605985379, alpha3=69713.71366346959):
    
    mse1 = F.mse_loss(pred, gt)
    mse2 = F.mse_loss(get_divergence(pred[0], pred[1]), get_divergence(gt[0], gt[1]))
    mse3 = F.mse_loss(get_vorticity(pred[0], pred[1]), get_vorticity(gt[0], gt[1]))

    return alpha1 * mse1 + alpha2 * mse2 + alpha3 * mse3

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
    n_channels = original.shape[0]

    for ch in range(n_channels):
        pearson_coeffs.append(stats.pearsonr(original[0, :, :].flatten(), predicted[0, :, :].flatten()))

    return pearson_coeffs

def get_all_metrics(original, predicted, is_divergence=False):
    """
    Calculate metrics for either velocity fields (2 channels) or divergence/vorticity (1 channel)
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
        orig_0 = original[0, :, :].cpu().numpy()
        orig_1 = original[1, :, :].cpu().numpy()
        pred_0 = predicted[0, :, :].cpu().numpy()
        pred_1 = predicted[1, :, :].cpu().numpy()

        slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(orig_0.flatten(), pred_0.flatten())
        slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(orig_1.flatten(), pred_1.flatten())
        pearson0, pearson1 = calculate_correlation(original, predicted)
        pearson0 = pearson0.statistic
        pearson1 = pearson1.statistic

        return {
            "mse": mse_l.item(), 
            "rmse": rmse_l.item(),
            "pearson_vx": pearson0.item(),
            "pearson_vy": pearson1.item(),
            "slope_vx": slope_x,
            "slope_vy": slope_y
        }