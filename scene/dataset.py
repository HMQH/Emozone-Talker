from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov


import cv2

import sys

def create_transformation_matrix(R, T):
    T_homogeneous = np.hstack((R, T.reshape(-1, 1)))  # Concatenate R and T horizontally
    T_homogeneous = np.vstack((T_homogeneous, [0, 0, 0, 1]))  # Add the last row for homogeneous coordinates
    return T_homogeneous


class FourDGSdataset(Dataset):
    """
    4D高斯点云数据集类 (4D Gaussian Splatting Dataset)
    
    核心功能：
        将CameraInfo列表转换为PyTorch Dataset，负责数据加载和格式转换。
        它实现了延迟加载（lazy loading），只在需要时才加载图像数据。
    
    主要职责：
        1. 接收CameraInfo列表（来自SceneInfo）
        2. 在__getitem__中实际加载图像数据（延迟加载）
        3. 转换数据格式：从numpy转换为torch.Tensor
        4. 创建Camera对象：用于渲染和训练
    
    为什么需要这个类？
        - SceneInfo只包含元数据（路径、参数），图像数据可能为None
        - 训练需要torch.Tensor格式，而不是numpy格式
        - 需要支持PyTorch的DataLoader（批量加载、随机采样等）
        - 渲染和训练需要Camera对象，而不是CameraInfo
    
    数据流：
        SceneInfo (CameraInfo列表) → FourDGSdataset → Camera对象 → 训练使用
    """
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        aud=None
    ):
        """
        初始化数据集
        
        参数:
            dataset (list): CameraInfo对象列表（来自SceneInfo）
            args: 配置参数对象
            dataset_type (str): 数据集类型（"ER-NeRF"等）
            aud: 音频数据（当前未使用）
        
        注意:
            dataset是CameraInfo列表，包含元数据（路径、参数等）
            图像数据会在__getitem__中延迟加载
        """
        self.dataset = dataset  # CameraInfo列表
        self.args = args
        self.dataset_type = dataset_type
    
    def __getitem__(self, index):
        """
        获取指定索引的数据项
        
        参数:
            index (int): 数据项的索引
        
        返回:
            Camera: Camera对象，包含渲染和训练所需的所有信息
        
        工作流程:
            1. 从CameraInfo列表中获取指定索引的CameraInfo
            2. 加载图像数据（如果还没有加载）：从磁盘读取图像文件
            3. 转换数据格式：从numpy转换为torch.Tensor，从(H,W,C)转换为(C,H,W)
            4. 处理其他数据：躯干图像、背景图像、掩码等
            5. 创建Camera对象并返回
        
        延迟加载（Lazy Loading）:
            - 图像数据只在__getitem__被调用时才加载
            - 节省内存：只加载当前批次需要的图像
            - 支持大数据集：可以处理超出内存的数据集
        """
        if self.dataset_type != "PanopticSports":
            # ========== 步骤1：获取CameraInfo ==========
            caminfo = self.dataset[index]  # 从CameraInfo列表中获取指定索引的CameraInfo
            
            # ========== 步骤2：提取相机参数 ==========
            R = caminfo.R  # (3, 3) - 旋转矩阵
            T = caminfo.T  # (3,) - 平移向量
            FovX = caminfo.FovX  # 水平视场角
            FovY = caminfo.FovY  # 垂直视场角
            trans = caminfo.trans.cpu().numpy()  # 3DMM平移参数

            # ========== 步骤3：加载完整图像（延迟加载） ==========
            # 如果图像数据还没有加载（为None），从磁盘读取
            mask = caminfo.mask
            full_image = caminfo.full_image
            if full_image is None:
                # 从磁盘读取图像文件
                full_image = cv2.imread(caminfo.full_image_path, cv2.IMREAD_UNCHANGED)
                # 转换颜色空间：BGR → RGB
                full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
                # 转换为torch.Tensor格式，并归一化到[0,1]
                # permute(2,0,1): 从(H, W, C)转换为(C, H, W)（PyTorch格式）
                full_image = torch.from_numpy(full_image).permute(2,0,1).float() / 255.0
                
            # ========== 步骤4：加载躯干图像（延迟加载） ==========
            torso_image = caminfo.torso_image
            if torso_image is None:
                # 从磁盘读取躯干图像（包含透明通道）
                torso_image = cv2.imread(caminfo.torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                # 转换颜色空间：BGRA → RGBA
                torso_image = cv2.cvtColor(torso_image, cv2.COLOR_BGRA2RGBA)
                # 归一化到[0,1]
                torso_image = torso_image.astype(np.float32) / 255 # [H, W, 4]
                # 转换为torch.Tensor格式
                torso_image = torch.from_numpy(torso_image)
                # permute(2, 0, 1): 从(H, W, 4)转换为(4, H, W)（PyTorch格式）
                torso_image = torso_image.permute(2, 0, 1)

            # ========== 步骤5：加载背景图像（延迟加载） ==========
            bg_image = caminfo.bg_image
            if bg_image is None:
                # 从磁盘读取背景图像
                bg_img = cv2.imread(caminfo.bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
                # 如果尺寸不匹配，调整大小
                if bg_img.shape[0] != caminfo.height or bg_img.shape[1] != caminfo.width:
                    bg_img = cv2.resize(bg_img, (caminfo.width, caminfo.height), interpolation=cv2.INTER_AREA)
                # 转换颜色空间：BGR → RGB
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                # 转换为torch.Tensor格式，并归一化到[0,1]
                bg_image = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
            
            # ========== 步骤6：处理掩码 ==========
            seg = caminfo.mask
            if seg is None:
                # 从磁盘读取掩码图像
                seg = cv2.imread(caminfo.mask_path)
            # 提取头部掩码：在语义分割中，头部区域用红色表示(R=255, G=0, B=0)
            head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            
            # ========== 步骤7：合成背景+躯干图像 ==========
            # bg_w_torso = 躯干图像 * 透明度 + 背景图像 * (1 - 透明度)
            bg_w_torso = torso_image[:3,...] * torso_image[3:,...] + bg_image * (1-torso_image[3:,...])
                        
            # ========== 步骤8：提取面部区域边界框 ==========
            face_rect = caminfo.face_rect  # 脸部边界框
            lhalf_rect = caminfo.lhalf_rect  # 左半脸边界框
            eye_rect = caminfo.eye_rect  # 眼部边界框
            lips_rect = caminfo.lips_rect  # 嘴唇边界框
            
            # ========== 步骤9：创建Camera对象 ==========
            # 将所有信息封装成Camera对象，用于渲染和训练
            return Camera(
                colmap_id=index,
                R=R, T=T,  # 相机外参
                FoVx=FovX, FoVy=FovY,  # 视场角
                gt_image=full_image,  # 真实图像（torch.Tensor格式）
                head_mask=head_mask,  # 头部掩码
                bg_image=bg_image,  # 背景图像
                image_name=f"{index}",
                uid=index,
                data_device=torch.device("cuda"),
                aud_f=caminfo.aud_f,  # 音频特征
                eye_f=caminfo.eye_f,  # 眼部特征
                au_au1_f=getattr(caminfo, "au_au1", None),   # AU1
                au_au2_f=getattr(caminfo, "au_au2", None),   # AU2
                au_au4_f=getattr(caminfo, "au_au4", None),   # AU4
                au_eye_f=getattr(caminfo, "au_eye", None),    # 眼部AU组
                au_nose_f=getattr(caminfo, "au_nose", None),  # 鼻子AU组
                face_rect=face_rect,  # 脸部边界框
                lhalf_rect=lhalf_rect,  # 左半脸边界框
                eye_rect=eye_rect,  # 眼部边界框
                lips_rect=lips_rect,  # 嘴唇边界框
                bg_w_torso=bg_w_torso,  # 背景+躯干图像
                aud_id=getattr(caminfo, "aud_id", 0)  # 3.15新添加 同步损失
            )
            
        else:
            return self.dataset[index]
    
    def __len__(self):
        """
        返回数据集的大小
        
        返回:
            int: CameraInfo列表的长度（相机数量）
        
        用途:
            - PyTorch的DataLoader需要知道数据集的大小
            - 用于随机采样和批量加载
        """
        return len(self.dataset)
