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

def div_vor_loss(pred, gt, alpha1=1.0, alpha2=72227.87605985379, alpha3=69713.71366346959, normalize=False):
    """
    Custom loss function to take into account divergence and vorticity - 
    compute a combined loss of MSE for velocity, divergence and vorticity.
    
    Args:
        pred: Predicted velocity field (2-channel tensor)
        gt: Ground truth velocity field (2-channel tensor)
        alpha1: Weight for velocity MSE
        alpha2: Weight for divergence MSE
        alpha3: Weight for vorticity MSE
    Returns:
        Combined loss value and individual MSE components
    """
    mse1 = F.mse_loss(pred, gt)

    if normalize:
        v_std = 230181.609375
    else:
        v_std = 1.0

    div_pred = get_divergence(pred[0], pred[1]) * v_std
    div_gt = get_divergence(gt[0], gt[1]) * v_std
    vor_pred = get_vorticity(pred[0], pred[1]) * v_std
    vor_gt = get_vorticity(gt[0], gt[1]) * v_std

    # mse2 = F.mse_loss(div_pred, div_gt)
    # mse3 = F.mse_loss(vor_pred, vor_gt)

    mse2 = F.l1_loss(div_pred, div_gt, reduction='mean')
    mse3 = F.l1_loss(vor_pred, vor_gt, reduction='mean')

    return alpha1 * mse1 + (alpha2 * mse2 + alpha3 * mse3), alpha1 * mse1, alpha2 * mse2, alpha3 * mse3

def div_vor_loss_norm(pred, gt, mean_gt_div, std_gt_div, mean_gt_vor, std_gt_vor):
    """
    Custom loss function to take into account divergence and vorticity - 
    compute a combined loss of MSE for velocity, divergence and vorticity.
    
    Args:
        pred: Predicted velocity field (2-channel tensor)
        gt: Ground truth velocity field (2-channel tensor)
        mean_gt_div: Mean of the ground truth divergence for normalization
        std_gt_div: Standard deviation of the ground truth divergence for normalization
        mean_gt_vor: Mean of the ground truth vorticity for normalization
        std_gt_vor: Standard deviation of the ground truth vorticity for normalization
    Returns:
        Combined loss value and individual MSE components
    """
    mse1 = F.mse_loss(pred, gt)


    div_pred = get_divergence(pred[0], pred[1])
    div_gt = get_divergence(gt[0], gt[1])
    vor_pred = get_vorticity(pred[0], pred[1])
    vor_gt = get_vorticity(gt[0], gt[1])

    div_pred = (div_pred - mean_gt_div) / std_gt_div
    div_gt = (div_gt - mean_gt_div) / std_gt_div
    vor_pred = (vor_pred - mean_gt_vor) / std_gt_vor
    vor_gt = (vor_gt - mean_gt_vor) / std_gt_vor

    mse2 = F.mse_loss(div_pred, div_gt) 
    mse3 = F.mse_loss(vor_pred, vor_gt) 
    return  mse1 + mse2 + mse3, mse1, mse2, mse3

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