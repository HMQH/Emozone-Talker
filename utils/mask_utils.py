import torch
import numpy as np

def generate_3d_point_masks_from_coords(points_3d, method='quantile'):
    """
    从 3D 点坐标生成区域 mask（路线 1：3D 坐标阈值）
    假设 y 轴是上下方向（后续需根据验证结果调整索引）
    
    参数:
        points_3d: (B, N, 3) 或 (N, 3) torch tensor
        method: 'quantile' (推荐) 或 'fixed'
    
    返回:
        upper_face_mask_3d: bool tensor
        mouth_mask_3d: bool tensor
    """
    # 统一处理成 (N, 3)
    if points_3d.dim() == 3:
        # 取第一个 batch 的点作为参考（假设所有 batch 点位置一致）
        pts = points_3d[0]
    else:
        pts = points_3d

    # 获取坐标轴（假设索引 1 是上下方向，索引 0 是左右方向）
    # 如发现上下轴不对，可将 y_coords 改为 pts[:, 2]
    y_coords = pts[:, 1]
    x_coords = pts[:, 0]
    
    if method == 'quantile':
        y_min, y_max = y_coords.min(), y_coords.max()
        y_range = y_max - y_min
        x_min, x_max = x_coords.min(), x_coords.max()
        x_range = x_max - x_min
        x_center = 0.5 * (x_min + x_max)
        
        # 上半脸（眉/眼/鼻）：避免只落在头发最上方，用一个“带宽”而不是纯 top-40%
        # 经验：取 [0.35, 0.80] 区间（可按你的数据微调）
        upper_low = y_min + 0.35 * y_range
        upper_high = y_min + 0.80 * y_range
        upper_face_mask_3d = (y_coords > upper_low) & (y_coords < upper_high)
        
        # 嘴部：收窄 y 区间，并加左右居中窗口，避免覆盖到脸颊
        # 经验：y ∈ [0.15, 0.30] 且 |x-xc| < 0.22*x_range（可微调）
        mouth_low = y_min + 0.15 * y_range
        mouth_high = y_min + 0.30 * y_range
        mouth_center_window = (torch.abs(x_coords - x_center) < (0.22 * x_range))
        mouth_mask_3d = (y_coords > mouth_low) & (y_coords < mouth_high) & mouth_center_window
        
    else:
        # 固定阈值方案 (备选)
        upper_face_mask_3d = (y_coords > 0.1)
        mouth_mask_3d = (y_coords > -0.15) & (y_coords < -0.05)
        
    return upper_face_mask_3d, mouth_mask_3d
