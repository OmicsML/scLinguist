import torch

def PearsonCorr1d(y_true, y_pred):
    assert len(y_true.shape) == 1
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.mean(torch.sum(y_true_c * y_pred_c) / (torch.sqrt(torch.sum(y_true_c * y_true_c)) + 1e-8)
                        / (torch.sqrt(torch.sum(y_pred_c * y_pred_c)) + 1e-8))
    return pearson

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the multi-scale Gaussian RBF kernel matrix between source and target.

    Args:
        source (Tensor): Source domain data of shape (n_samples, features)
        target (Tensor): Target domain data of shape (m_samples, features)
        kernel_mul (float): Multiplier to scale the kernel bandwidth
        kernel_num (int): Number of different Gaussian kernels to compute
        fix_sigma (float, optional): If provided, use this fixed bandwidth instead of computing it

    Returns:
        Tensor: Combined Gaussian kernel matrix of shape (n + m, n + m)
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)

    # Compute pairwise L2 distances
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))

    # Batch-wise distance computation to save memory
    batch_size = 200
    num_window = total0.size(0) // batch_size + 1
    L2_dis = []
    for i in range(num_window):
        diff = (total0[i * batch_size:(i + 1) * batch_size].cuda() - 
                total1[i * batch_size:(i + 1) * batch_size].cuda())
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.concatenate(L2_dis, dim=0)

    # Compute bandwidth(s)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Compute multi-scale Gaussian kernels
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)


def mmd_rbf(y_true, y_pred, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between y_true and target using RBF kernel.

    Args:
        y_true (Tensor): y_true data of shape (n_samples, features)
        y_pred (Tensor): y_pred domain data of shape (n_samples, features)
        kernel_mul (float): Multiplier to scale the kernel bandwidth
        kernel_num (int): Number of RBF kernels used for multi-scale MMD
        fix_sigma (float, optional): Fixed bandwidth value

    Returns:
        Tensor: Scalar MMD loss value
    """
    batch_size = y_true.size(0)
    kernels = gaussian_kernel(y_true, y_pred,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)

    # Partition the combined kernel matrix
    XX = kernels[:batch_size, :batch_size]   # Source-source
    YY = kernels[batch_size:, batch_size:]   # Target-target
    XY = kernels[:batch_size, batch_size:]   # Source-target
    YX = kernels[batch_size:, :batch_size]   # Target-source

    # Compute MMD loss
    loss = torch.mean(XX + YY - XY - YX)
    return loss
