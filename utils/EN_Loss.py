import torch
import numpy as np
# Effective Number of Samples
def get_en_weights(num_samples, beta=0.99, device='cpu'):
    """
    根据 Effective Number of Samples 计算类别权重

    Args:
        num_samples (list or np.array): 每个类别的样本数量
        beta (float): 衰减系数，常用 0.99~0.9999
        device (str): 'cpu' 或 'cuda'

    Returns:
        torch.Tensor: 归一化后的类别权重，可直接用于 CrossEntropyLoss
    """
    num_samples = np.array(num_samples)
    effective_num = 1.0 - np.power(beta, num_samples)
    weights = (1.0 - beta) / effective_num
    # 归一化，使权重和类别数相等
    weights = weights / np.sum(weights) * len(num_samples)
    return torch.tensor(weights, dtype=torch.float32, device=device)
