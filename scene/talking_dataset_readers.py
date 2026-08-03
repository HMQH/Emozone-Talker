#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
from scene.utils import get_audio_features

import cv2

from scene.dataset_readers import read_timeline, getNerfppNorm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    full_image: np.array
    full_image_path: str
    image_name: str
    width: int
    height: int
    torso_image: np.array
    torso_image_path: str
    bg_image: np.array
    bg_image_path: str
    mask: np.array
    mask_path: str
    trans: np.array
    face_rect: list
    lhalf_rect: list
    aud_f: torch.FloatTensor
    eye_f: np.array
    #2.1修改
    au_au1: np.array   # AU1，形状(1,)或None，从JAANet提取
    au_au2: np.array   # AU2，形状(1,)或None，从JAANet提取
    au_au4: np.array   # AU4，形状(1,)或None，从JAANet提取
    #2.1
    au_eye: np.array   # 眼部AU组，形状(1,)或None，从JAANet提取：AU6（缺少AU5和AU7）
    au_nose: np.array  # 鼻子AU组，形状(1,)或None，从JAANet提取：AU9
    eye_rect: list
    lips_rect: list
    aud_id: int = 0  # 3.15新添加：同步损失：全局帧索引，用于从 wav 提取对应音频
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    custom_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def read_and_extract_head(ori_image_path):
    ori_image = cv2.imread(ori_image_path, cv2.IMREAD_UNCHANGED)
    seg = cv2.imread(ori_image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
    head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
    #0126
    # seg = cv2.imread(
    #     ori_image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'),
    #     cv2.IMREAD_UNCHANGED
    # )
    # if seg.ndim == 3:
    #     seg = seg[..., 0]
    # head_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]
    # head_mask = np.isin(seg, head_ids)
    #0126

    # Create an empty image with the same shape as the original image
    # head_image = np.zeros_like(ori_image)

    # Apply the mask to the original image to extract the head part
    # head_image[head_mask] = ori_image[head_mask]
    
    return ori_image, head_mask
    

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        # fovy = fovx
        FovY = fovy 
        FovX = fovx
        # 创建CameraInfo对象（视频渲染轨迹，AU相关字段置为None）
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            full_image=image, full_image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            torso_image=None, torso_image_path=None, bg_image=None, bg_image_path=None,
                            mask=None, mask_path=None, trans=np.zeros(3), face_rect=None, lhalf_rect=None,
                            #2.1修改
                            aud_f=None, eye_f=None, au_au1=None, au_au2=None, au_au4=None, au_eye=None, au_nose=None,
                            #2.1
                            eye_rect=None, lips_rect=None, aud_id=0))
    return cam_infos

def euler2rot2(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat(
        (
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ),
        2,
    )
    rot_y = torch.cat(
        (
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ),
        2,
    )
    rot_z = torch.cat(
        (
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1),
        ),
        2,
    )
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))
def normalize_au_values(values, max_value=5, auto_detect=True, normalized_upper=1.2):
    """
    将 AU 值归一化到 [0, 1]。

    - auto_detect=True 时自动兼容两种输入：
      1) 已归一化的 [0,1]（或轻微浮动） -> 直接裁剪到 [0,1]
      2) OpenFace 强度 [0,5] -> 先裁剪到 [0,5] 再除以 5
    - 缺失值请在进入此函数前填零。
    """
    arr = np.asarray(values, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    if auto_detect:
        vmax = float(valid.max())
        if vmax <= normalized_upper:
            return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return (np.clip(arr, 0.0, float(max_value)) / float(max_value)).astype(np.float32)

#0130开始
def extract_au_from_video_jaanet(
    video_path: str,
    num_frames: int,
    jaanet_checkpoints_dir: str = None,
    jaanet_dataset: str = "DISFA",
    jaanet_device: str = "cuda",
    jaanet_version: str = "v1",
    frame_stride: int = 1,
    cache_dir: str = None,
):
    # JAANet is optional for inference from a complete AU CSV. Import it only
    # when video-based AU extraction is explicitly requested.
    from utils.jaanet_extractor import JaaNetExtractor
    """
    从视频文件使用 JAANet 提取 AU 值。
    
    参数:
        video_path: 视频文件路径（.mp4）
        num_frames: 期望的帧数（用于对齐）
        jaanet_checkpoints_dir: JAANet 权重目录，如果为 None 则使用默认路径
        jaanet_dataset: "BP4D" 或 "DISFA"
        jaanet_device: "cuda" 或 "cpu"
        jaanet_version: "v1" 或 "v2"
        frame_stride: 抽帧间隔
        cache_dir: 缓存目录，如果为 None 则使用视频文件所在目录下的 .au_cache 目录
    
    返回:
        au_features: dict，包含 "au1", "au2", "au4", "eye", "nose" 五个组
            - "au1": [num_frames, 1]，包含 AU1
            - "au2": [num_frames, 1]，包含 AU2
            - "au4": [num_frames, 1]，包含 AU4
            - "eye": [num_frames, 1]，包含 AU6
            - "nose": [num_frames, 1]，包含 AU9
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 生成缓存文件路径
    import hashlib
    video_abs_path = os.path.abspath(video_path)
    video_hash = hashlib.md5(video_abs_path.encode()).hexdigest()[:8]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(video_path), ".au_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 缓存文件名包含视频信息、数据集类型、版本和期望帧数
    cache_filename = f"au_{video_name}_{video_hash}_{jaanet_dataset}_{jaanet_version}_f{num_frames}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # 尝试加载缓存
    if os.path.exists(cache_path):
        try:
            print(f"[INFO] 发现 AU 缓存文件，正在加载: {cache_path}")
            cached_data = np.load(cache_path)
            #2.1修改
            cached_au1 = cached_data["au1"]
            cached_au2 = cached_data["au2"]
            cached_au4 = cached_data["au4"]
            #2.1
            cached_eye = cached_data["eye"]
            cached_nose = cached_data["nose"]
            
            # 验证缓存数据的形状是否匹配
            #2.1修改
            if (cached_au1.shape[0] == num_frames and 
                cached_au2.shape[0] == num_frames and 
                cached_au4.shape[0] == num_frames and
                cached_eye.shape[0] == num_frames and 
                cached_nose.shape[0] == num_frames):
                print(f"[INFO] 成功加载 AU 缓存，跳过提取过程")
                return {
                    "au1": cached_au1,
                    "au2": cached_au2,
                    "au4": cached_au4,
                    "eye": cached_eye,
                    "nose": cached_nose,
                }
            else:
                print(f"[WARN] 缓存数据形状不匹配（期望 {num_frames} 帧，缓存为 {cached_au1.shape[0]} 帧），重新提取...")
            #2.1
        except Exception as e:
            print(f"[WARN] 加载 AU 缓存失败: {e}，重新提取...")
    
    # 如果未提供 checkpoints_dir，尝试使用默认路径
    if jaanet_checkpoints_dir is None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        default_checkpoints_dir = os.path.join(project_root, "PyTorch-JAANet", "checkpoints")
        if os.path.exists(default_checkpoints_dir):
            jaanet_checkpoints_dir = default_checkpoints_dir
        else:
            raise FileNotFoundError(
                f"未找到 JAANet 权重目录。请提供 jaanet_checkpoints_dir 参数，"
                f"或确保 PyTorch-JAANet/checkpoints 目录存在。"
            )
    else:
        # 验证路径是否为绝对路径，如果是相对路径则转换为绝对路径
        if not os.path.isabs(jaanet_checkpoints_dir):
            # 如果是相对路径，基于项目根目录解析
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            jaanet_checkpoints_dir = os.path.join(project_root, jaanet_checkpoints_dir)
        
        # 验证提供的路径是否存在
        if not os.path.exists(jaanet_checkpoints_dir):
            raise FileNotFoundError(
                f"提供的 JAANet 权重目录不存在: {jaanet_checkpoints_dir}\n"
                f"请检查路径是否正确，或使用 --jaanet_checkpoints_dir 参数指定正确的路径。\n"
                f"当前工作目录: {os.getcwd()}"
            )
        # 验证目录中是否包含必要的权重文件（至少检查一个）
        test_files = ["region_learning.pth", "align_net.pth"]
        found_any = any(os.path.exists(os.path.join(jaanet_checkpoints_dir, f)) for f in test_files)
        if not found_any:
            print(f"[WARN] 在 {jaanet_checkpoints_dir} 中未找到预期的权重文件（如 region_learning.pth），但继续尝试加载...")
    
    # 初始化 JAANet 提取器
    print(f"[INFO] 初始化 JAANet 提取器（dataset={jaanet_dataset}, version={jaanet_version}）...")
    au_extractor = JaaNetExtractor(
        checkpoints_dir=jaanet_checkpoints_dir,
        dataset=jaanet_dataset,
        device=jaanet_device,
        version=jaanet_version,
    )
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 视频信息: 总帧数={total_frames}, FPS={fps:.2f}")
    
    # 提取所有帧的 AU
    #2.1修改
    all_au_au1 = []   # AU1
    all_au_au2 = []   # AU2
    all_au_au4 = []   # AU4
    #2.1
    all_au_eye = []   # AU6
    all_au_nose = []  # AU9
    
    frame_idx = 0
    processed_count = 0
    
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="从视频提取AU (JAANet)", unit="frame")
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # 根据 frame_stride 决定是否处理这一帧
            if frame_idx % frame_stride == 0:
                # 转换为 RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # 转换为 [0, 1] 范围的 torch tensor [1, 3, H, W]
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(jaanet_device)
                
                # 提取 AU
                au_probs = au_extractor(frame_tensor)  # [1, K]
                au_probs = au_probs.squeeze(0).cpu().numpy()  # [K]
                
                #2.1修改
                if jaanet_dataset.upper() == "BP4D":
                    # BP4D 索引: 0=AU1, 1=AU2, 2=AU4, 3=AU6, ...
                    au_au1_frame = np.array([au_probs[0]])  # AU1
                    au_au2_frame = np.array([au_probs[1]])  # AU2
                    au_au4_frame = np.array([au_probs[2]])  # AU4
                    au_eye_frame = np.array([au_probs[3]])  # AU6
                    au_nose_frame = np.array([0.0])  # BP4D 没有 AU9，用 0 填充
                elif jaanet_dataset.upper() == "DISFA":
                    # DISFA 索引: 0=AU1, 1=AU2, 2=AU4, 3=AU6, 4=AU9, ...
                    au_au1_frame = np.array([au_probs[0]])  # AU1
                    au_au2_frame = np.array([au_probs[1]])  # AU2
                    au_au4_frame = np.array([au_probs[2]])  # AU4
                    au_eye_frame = np.array([au_probs[3]])  # AU6
                    au_nose_frame = np.array([au_probs[4]])  # AU9
                else:
                    raise ValueError(f"不支持的数据集类型: {jaanet_dataset}，应为 BP4D 或 DISFA")
        
                all_au_au1.append(au_au1_frame)
                all_au_au2.append(au_au2_frame)
                all_au_au4.append(au_au4_frame)
                all_au_eye.append(au_eye_frame)
                all_au_nose.append(au_nose_frame)
                processed_count += 1
                #2.1
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
    
    print(f"[INFO] 成功提取 {processed_count} 帧的 AU 值")
    
    # 转换为 numpy 数组
    #2.1修改
    if len(all_au_au1) > 0:
        au_au1_array = np.stack(all_au_au1, axis=0)  # [processed_count, 1]
        au_au2_array = np.stack(all_au_au2, axis=0)  # [processed_count, 1]
        au_au4_array = np.stack(all_au_au4, axis=0)  # [processed_count, 1]
        au_eye_array = np.stack(all_au_eye, axis=0)     # [processed_count, 1]
        au_nose_array = np.stack(all_au_nose, axis=0)   # [processed_count, 1]
    else:
        # 如果没有提取到任何帧，返回全零数组
        au_au1_array = np.zeros((num_frames, 1))
        au_au2_array = np.zeros((num_frames, 1))
        au_au4_array = np.zeros((num_frames, 1))
        au_eye_array = np.zeros((num_frames, 1))
        au_nose_array = np.zeros((num_frames, 1))
        return {
            "au1": au_au1_array,
            "au2": au_au2_array,
            "au4": au_au4_array,
            "eye": au_eye_array,
            "nose": au_nose_array,
        }
    #2.1
    # 对齐到期望的帧数
    #2.1修改
    if processed_count > num_frames:
        # 截断
        print(f"[INFO] 提取的帧数 ({processed_count}) > 期望帧数 ({num_frames})，截断...")
        au_au1_array = au_au1_array[:num_frames]
        au_au2_array = au_au2_array[:num_frames]
        au_au4_array = au_au4_array[:num_frames]
        au_eye_array = au_eye_array[:num_frames]
        au_nose_array = au_nose_array[:num_frames]
    elif processed_count < num_frames:
        # 填充（使用最后一帧的值）
        pad_len = num_frames - processed_count
        print(f"[INFO] 提取的帧数 ({processed_count}) < 期望帧数 ({num_frames})，填充 {pad_len} 帧...")
        last_au1 = au_au1_array[-1:]
        last_au2 = au_au2_array[-1:]
        last_au4 = au_au4_array[-1:]
        last_eye = au_eye_array[-1:]
        last_nose = au_nose_array[-1:]
        au_au1_array = np.concatenate([au_au1_array, np.repeat(last_au1, pad_len, axis=0)], axis=0)
        au_au2_array = np.concatenate([au_au2_array, np.repeat(last_au2, pad_len, axis=0)], axis=0)
        au_au4_array = np.concatenate([au_au4_array, np.repeat(last_au4, pad_len, axis=0)], axis=0)
        au_eye_array = np.concatenate([au_eye_array, np.repeat(last_eye, pad_len, axis=0)], axis=0)
        au_nose_array = np.concatenate([au_nose_array, np.repeat(last_nose, pad_len, axis=0)], axis=0)
    #2.1修改
    au_features = {
        "au1": au_au1_array,  # [num_frames, 1]
        "au2": au_au2_array,  # [num_frames, 1]
        "au4": au_au4_array,  # [num_frames, 1]
        "eye": au_eye_array,    # [num_frames, 1]
        "nose": au_nose_array,  # [num_frames, 1]
    }
    
    # 保存到缓存
    try:
        np.savez(cache_path, 
                 au1=au_au1_array, 
                 au2=au_au2_array,
                 au4=au_au4_array,
                 eye=au_eye_array, 
                 nose=au_nose_array)
        print(f"[INFO] AU 值已保存到缓存: {cache_path}")
    except Exception as e:
        print(f"[WARN] 保存 AU 缓存失败: {e}，但继续执行...")
    
    return au_features
#0130结束

def readCamerasFromTracksTransforms(path, meshfile, transformsfile, aud_features, eye_features, au_features,
                                    extension=".jpg", mapper = {}, preload=False, custom_aud =None, use_custom_au=False, args=None):
    cam_infos = []
    mesh_path = os.path.join(path, meshfile)
    track_params = torch.load(mesh_path)
    trans_infos = torch.load(mesh_path)["trans"]
    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        
    contents["fl_x"] = contents["focal_len"]
    contents["fl_y"] = contents["focal_len"]
    contents["w"] = contents["cx"] * 2
    contents["h"] = contents["cy"] * 2

    fovx = focal2fov(contents['fl_x'],contents['w'])
    fovy = focal2fov(contents['fl_y'],contents['h'])
    frames = contents["frames"]
    if custom_aud:
        if len(frames) == 0:
            raise ValueError(f"No camera frames found in {transformsfile}")
        target_count = int(aud_features.shape[0])
        frames = [frames[i % len(frames)] for i in range(target_count)]
    f_path = os.path.join(path, "ori_imgs")
    
    FovY = fovy 
    FovX = fovx
    
    # background_image
    bg_image_path = os.path.join(path, "bc.jpg")
    bg_img = cv2.imread(bg_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
    if bg_img.shape[0] != contents["h"] or bg_img.shape[1] != contents["w"]:
        bg_img = cv2.resize(bg_img, (contents["w"], contents["h"]), interpolation=cv2.INTER_AREA)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = torch.from_numpy(bg_img).permute(2,0,1).float() / 255.0
        
    if custom_aud:
        auds = aud_features
    else:    
        auds = [aud_features[min(frame['aud_id'], aud_features.shape[0] - 1)] for frame in frames]
        auds = torch.stack(auds, dim=0)
        
    for idx, frame in enumerate(frames): # len(frames): 7272
        
        cam_name = os.path.join(f_path, str(frame["img_id"]) + extension)
        aud_feature = get_audio_features(auds, att_mode = 2, index = idx)     
        
        
        # Camera Codes
        euler = track_params["euler"][frame["img_id"]]
        R = euler2rot(euler.unsqueeze(0))
        
        flip_rot = torch.tensor(
            [[-1,  0,  0],  # This flips the X axis
            [ 0,  1,  0],  # Y axis remains the same
            [ 0,  0, 1]], # This flips the Z axis, maintaining the right-hand rule
            dtype=R.dtype,
            device=R.device
        ).view(1, 3, 3)
        # flip_rot = flip_rot.expand_as(R)  # Make sure it has the same batch size as R

        # Apply the flip rotation by matrix multiplication
        # Depending on your convention, you might need to apply the flip before or after the original rotation.
        # Use torch.matmul(flip_rot, R) if the flip should be applied globally first,
        # or torch.matmul(R, flip_rot) if the flip should be applied in the camera's local space.
        R = torch.matmul(flip_rot, R)
        R = R.squeeze(0).cpu().numpy()
        T = track_params["trans"][frame["img_id"]].unsqueeze(0).cpu().numpy()
        
        R = -np.transpose(R)
        T = -T
        T[:, 0] = -T[:, 0] 

        # Get Iamges for Facial 
        image_name = Path(cam_name).stem

        full_image_path = cam_name
        torso_image_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
        mask_path = cam_name.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        
        # Landmark and extract face
        lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]

        lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
        xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
        ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
        face_rect = [xmin, xmax, ymin, ymax]
        lhalf_rect = [lh_xmin, lh_xmax, ymin, ymax]
        
        # Eye Area and Eye Rect
        if use_custom_au:
            eye_area = eye_features[idx]
        else:
            eye_area = eye_features[frame['img_id']]
        # 兼容 AU45 输入为 0-5（OpenFace）或 0-1（手工/JAANet）
        eye_area = normalize_au_values(
            eye_area,
            max_value=5,
            auto_detect=use_custom_au,
        )
        # 选择性加载其他AU组,未启用时返回None，渲染阶段再用全零填充
        use_au_au1 = getattr(args, "use_au_au1", False) if args is not None else False
        use_au_au2 = getattr(args, "use_au_au2", False) if args is not None else False
        use_au_au4 = getattr(args, "use_au_au4", False) if args is not None else False
        use_au_eye = getattr(args, "use_au_eye", False) if args is not None else False
        use_au_nose = getattr(args, "use_au_nose", False) if args is not None else False
        #0129
        # 从 JAANet 提取的 AU 已经是 [0, 1] 范围，不需要归一化
        frame_id = idx if use_custom_au else frame["img_id"]
        au1_au = au_features["au1"][frame_id] if use_au_au1 else None
        au2_au = au_features["au2"][frame_id] if use_au_au2 else None
        au4_au = au_features["au4"][frame_id] if use_au_au4 else None
        eye_group_au = au_features["eye"][frame_id] if use_au_eye else None
        nose_au = au_features["nose"][frame_id] if use_au_nose else None
        #0129
        xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
        ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
        eye_rect = [xmin, xmax, ymin, ymax]
        
        # Finetune Lip Area
        lips = slice(48, 60)
        xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
        ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        l = max(xmax - xmin, ymax - ymin) // 2
        xmin = max(0, cx - l)
        xmax = min(contents["h"], cx + l)
        ymin = max(0, cy - l)
        ymax = min(contents["w"], cy + l)

        lips_rect = [xmin, xmax, ymin, ymax]
        
        if preload:
            ori_image = cv2.imread(cam_name, cv2.IMREAD_UNCHANGED)
            seg = cv2.imread(mask_path)
            head_mask = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
            #0126
            # seg = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            # if seg.ndim == 3:
            #     seg = seg[..., 0]
            # head_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]
            # head_mask = np.isin(seg, head_ids)
            #0126
            ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            ori_image = torch.from_numpy(ori_image).permute(2,0,1).float() / 255.0
            
            # torso images 
            torso_img = cv2.imread(torso_image_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
            torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
            torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]
            
        else:
            ori_image = None
            torso_img = None
            head_mask = None
            seg = None
        

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, full_image=ori_image, full_image_path=full_image_path,
                        image_name=image_name, width=contents["w"], height=contents["h"],
                        torso_image=torso_img, torso_image_path=torso_image_path, bg_image=bg_img, bg_image_path=bg_image_path,
                        mask=seg, mask_path=mask_path, trans=trans_infos[frame["img_id"]],
                        face_rect=face_rect, lhalf_rect=lhalf_rect, aud_f=aud_feature, eye_f=eye_area,
                        au_au1=au1_au, au_au2=au2_au, au_au4=au4_au, au_eye=eye_group_au, au_nose=nose_au,
                        eye_rect=eye_rect, lips_rect=lips_rect, aud_id=frame.get("aud_id", idx)))
    return cam_infos     





def readTalkingPortraitDatasetInfo(path, white_background, eval, extension=".jpg",custom_aud=None, custom_au=None, args=None):
    # Audio Information
    aud_features = np.load(os.path.join(path, 'aud_ds.npy'))
    aud_features = torch.from_numpy(aud_features)

    # support both [N, 16] labels and [N, 16, K] logits
    if len(aud_features.shape) == 3:
        aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
    else:
        raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')


    print(f'[INFO] load aud_features: {aud_features.shape}')
    
    # load action units
    import pandas as pd
    au_blink_info = pd.read_csv(os.path.join(path, 'au.csv'))
    au_blink_info.columns = au_blink_info.columns.str.strip()

#0130开始
    # 眨眼 AU45 仍然一律从 au.csv 读取（与原实现保持一致）
    if 'AU45_r' not in au_blink_info.columns:
        raise KeyError(
            f"au.csv 中缺少 AU45_r 列（眨眼）。当前列名: {list(au_blink_info.columns)}\n"
            f"请确保 au.csv 由 OpenFace 等工具生成且包含 AU45_r 列。"
        )
    eye_features = au_blink_info['AU45_r'].values

    # 是否在预处理阶段启用 JAANet（由 --use_jaanet_preprocess 控制）
    use_jaanet_preprocess = getattr(args, "use_jaanet_preprocess", False) if args is not None else False

    if use_jaanet_preprocess:
        # ---------- 分支 A：使用 JAANet 从视频提取 AU ----------
        video_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
        if len(video_files) == 0:
            raise FileNotFoundError(f"在 {path} 中未找到 .mp4 视频文件")
        elif len(video_files) > 1:
            print(f"[WARN] 找到多个视频文件: {video_files}，使用第一个: {video_files[0]}")
        video_path = os.path.join(path, video_files[0])
        num_frames = len(eye_features)
        if args is not None:
            jaanet_checkpoints_dir = getattr(args, "jaanet_checkpoints_dir", None)
            jaanet_dataset = getattr(args, "jaanet_dataset", "DISFA")
            jaanet_device = getattr(args, "jaanet_device", "cuda")
            jaanet_version = getattr(args, "jaanet_version", "v1")
        else:
            jaanet_checkpoints_dir = None
            jaanet_dataset = "DISFA"
            jaanet_device = "cuda"
            jaanet_version = "v1"
        print(f"[INFO] 从视频提取 AU（使用 JAANet）: {video_path}")
        au_features = extract_au_from_video_jaanet(
            video_path=video_path,
            num_frames=num_frames,
            jaanet_checkpoints_dir=jaanet_checkpoints_dir,
            jaanet_dataset=jaanet_dataset,
            jaanet_device=jaanet_device,
            jaanet_version=jaanet_version,
        )
        print(f"[INFO] AU 提取完成: au1={au_features['au1'].shape}, au2={au_features['au2'].shape}, au4={au_features['au4'].shape}, eye={au_features['eye'].shape}, nose={au_features['nose'].shape}")
    else:
        # ---------- 分支 B：不使用 JAANet，仅使用 OpenFace 生成的 au.csv ----------
        num_frames = len(eye_features)

        def _get_au_col(df, name, default_zero=True):
            if name in df.columns:
                vals = df[name].values.reshape(-1, 1)
                return normalize_au_values(vals)
            if default_zero:
                return np.zeros((num_frames, 1), dtype=np.float32)
            raise KeyError(f"列 {name} 在 au.csv 中不存在")

        required_au_cols = ["AU01_r", "AU02_r", "AU04_r", "AU06_r", "AU09_r"]
        missing = [c for c in required_au_cols if c not in au_blink_info.columns]
        if missing:
            print(f"[WARN] au.csv 缺失 AU 列（已用 0 填充）: {missing}")

        au1_arr = _get_au_col(au_blink_info, "AU01_r", default_zero=True)
        au2_arr = _get_au_col(au_blink_info, "AU02_r", default_zero=True)
        au4_arr = _get_au_col(au_blink_info, "AU04_r", default_zero=True)
        eye_arr = _get_au_col(au_blink_info, "AU06_r", default_zero=True)
        nose_arr = _get_au_col(au_blink_info, "AU09_r", default_zero=True)

        au_features = {
            "au1": au1_arr,
            "au2": au2_arr,
            "au4": au4_arr,
            "eye": eye_arr,
            "nose": nose_arr,
        }
        print(f"[INFO] 使用 au.csv 中的 AU 值（未开启 JAANet 预处理）: "
              f"au1={au_features['au1'].shape}, au2={au_features['au2'].shape}, "
              f"au4={au_features['au4'].shape}, eye={au_features['eye'].shape}, "
              f"nose={au_features['nose'].shape}")
    ply_path = os.path.join(path, "fused.ply")
    mesh_path = os.path.join(path, "track_params.pt")
    
    timestamp_mapper, max_time = read_timeline(path)
    #读取验证集frames长度,用于自定义AU长度对齐
    with open(os.path.join(path, "transforms_val.json")) as json_file:
        val_contents = json.load(json_file)
        target_frame_count = len(val_contents["frames"])
    if custom_aud:
        custom_audio_path = os.path.join(path, custom_aud)
        target_frame_count = int(np.load(custom_audio_path, mmap_mode="r").shape[0])

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_train.json", aud_features, eye_features, au_features, extension, timestamp_mapper, preload = False, args=args)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_val.json", aud_features, eye_features, au_features, extension, timestamp_mapper, args=args)
    print("Generating Video Transforms")
    video_cam_infos = None 

    #处理自定义AU,并长度对齐
    use_custom_au = False
    eye_features_custom = None
    au_features_custom = None
    if custom_au:
        custom_au_path = os.path.join(path, custom_au)
        custom_au_blink_info = pd.read_csv(custom_au_path)
        
        # 清理列名（去除前后空格）
        custom_au_blink_info.columns = custom_au_blink_info.columns.str.strip()
        
        # 预取视频路径，供异常回退使用
        video_files_for_fallback = [f for f in os.listdir(path) if f.endswith('.mp4')]
        video_path_for_fallback = os.path.join(path, video_files_for_fallback[0]) if video_files_for_fallback else None
        #0312开始
        # 从 CSV 读取 AU45（眨眼），兼容 0-5（OpenFace）与 0-1（手工/模型输出）
        if 'AU45_r' in custom_au_blink_info.columns:
            eye_features_custom = custom_au_blink_info['AU45_r'].values.astype(np.float32)
            print(f"[INFO] 从自定义 CSV 读取 AU45（自动兼容 0-5/0-1）: {len(eye_features_custom)} 帧")
        #0312结束
        else:
            print(f"[WARN] 自定义 CSV 中缺少 AU45_r 列，从默认 au.csv 读取...")
            default_au_path = os.path.join(path, 'au.csv')
            if os.path.exists(default_au_path):
                default_au_blink_info = pd.read_csv(default_au_path)
                default_au_blink_info.columns = default_au_blink_info.columns.str.strip()
                eye_features_custom = default_au_blink_info['AU45_r'].values
                print(f"[INFO] 从默认 au.csv 读取 AU45: {len(eye_features_custom)} 帧")
            else:
                raise FileNotFoundError(
                    f"自定义 CSV 中缺少 AU45_r 列，且默认 au.csv 文件不存在: {default_au_path}\n"
                    f"请确保自定义 CSV 包含 AU45_r 列，或确保默认 au.csv 文件存在。"
                )
        #0312开始    
        # 从 CSV 读取其他 AU 值（au1, au2, au4, eye, nose）
        custom_num_frames = len(eye_features_custom)
        
        # 尝试从 CSV 读取 AU 值，如果不存在则使用默认值或从视频提取
        try:
            # 检查 CSV 中是否存在所需的 AU 列
            #2.1
            required_au_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU09_r']  # AU1, AU2, AU4, AU6, AU9
            
            # 检查所有必需的列是否存在
            #2.1
            missing_cols = [col for col in required_au_cols if col not in custom_au_blink_info.columns]
            
            if missing_cols:
                print(f"[WARN] CSV 文件中缺少以下 AU 列: {missing_cols}")
                print(f"[WARN] 将尝试从视频提取缺失的 AU 值...")
                video_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
                if len(video_files) == 0:
                    raise FileNotFoundError(f"在 {path} 中未找到 .mp4 视频文件（提取缺失 AU 需要）")
                video_path = os.path.join(path, video_files[0])
                # 如果缺少列，尝试从视频提取
                jaanet_checkpoints_dir = getattr(args, "jaanet_checkpoints_dir", None) if args else None
                jaanet_dataset = getattr(args, "jaanet_dataset", "DISFA") if args else "DISFA"
                jaanet_device = getattr(args, "jaanet_device", "cuda") if args else "cuda"
                jaanet_version = getattr(args, "jaanet_version", "v1") if args else "v1"
                
                print(f"[INFO] 从视频提取缺失的 AU（使用 JAANet）: {video_path}")
                au_features_from_video = extract_au_from_video_jaanet(
                    video_path=video_path,
                    num_frames=custom_num_frames,
                    jaanet_checkpoints_dir=jaanet_checkpoints_dir,
                    jaanet_dataset=jaanet_dataset,
                    jaanet_device=jaanet_device,
                    jaanet_version=jaanet_version,
                )
                
                # 从 CSV 读取存在的列，从视频补充缺失的列
                au_au1_list = []
                au_au2_list = []
                au_au4_list = []
                au_eye_list = []
                au_nose_list = []
                
                for i in range(custom_num_frames):
                    # AU1：优先从 CSV 读取（OpenFace 0-5，需归一化），缺失则从视频
                    if 'AU01_r' in custom_au_blink_info.columns:
                        au1_frame = normalize_au_values(np.array([custom_au_blink_info['AU01_r'].iloc[i]]), max_value=5)
                    else:
                        au1_frame = au_features_from_video['au1'][i]
                    
                    if 'AU02_r' in custom_au_blink_info.columns:
                        au2_frame = normalize_au_values(np.array([custom_au_blink_info['AU02_r'].iloc[i]]), max_value=5)
                    else:
                        au2_frame = au_features_from_video['au2'][i]
                    
                    if 'AU04_r' in custom_au_blink_info.columns:
                        au4_frame = normalize_au_values(np.array([custom_au_blink_info['AU04_r'].iloc[i]]), max_value=5)
                    else:
                        au4_frame = au_features_from_video['au4'][i]
                    
                    if 'AU06_r' in custom_au_blink_info.columns:
                        eye_frame = normalize_au_values(np.array([custom_au_blink_info['AU06_r'].iloc[i]]), max_value=5)
                    else:
                        eye_frame = au_features_from_video['eye'][i]
                    
                    if 'AU09_r' in custom_au_blink_info.columns:
                        nose_frame = normalize_au_values(np.array([custom_au_blink_info['AU09_r'].iloc[i]]), max_value=5)
                    else:
                        nose_frame = au_features_from_video['nose'][i]
                    
                    au_au1_list.append(au1_frame)
                    au_au2_list.append(au2_frame)
                    au_au4_list.append(au4_frame)
                    au_eye_list.append(eye_frame)
                    au_nose_list.append(nose_frame)
                
                au_features_custom = {
                    "au1": np.stack(au_au1_list, axis=0),
                    "au2": np.stack(au_au2_list, axis=0),
                    "au4": np.stack(au_au4_list, axis=0),
                    "eye": np.stack(au_eye_list, axis=0),
                    "nose": np.stack(au_nose_list, axis=0),
                }
                
            else:
                # 所有必需的列都存在，直接从 CSV 读取并自动兼容 0-5/0-1
                print(f"[INFO] 从 CSV 文件读取 AU 值（自动兼容 0-5/0-1）: {custom_au_path}")
                
                au_au1_array = custom_au_blink_info['AU01_r'].values.reshape(-1, 1)  # [num_frames, 1]
                au_au2_array = custom_au_blink_info['AU02_r'].values.reshape(-1, 1)
                au_au4_array = custom_au_blink_info['AU04_r'].values.reshape(-1, 1)
                au_eye_array = custom_au_blink_info['AU06_r'].values.reshape(-1, 1)
                au_nose_array = custom_au_blink_info['AU09_r'].values.reshape(-1, 1)
                
                # 自动兼容 OpenFace 0-5 与已归一化 0-1 输入，统一到 [0,1]
                au_au1_array = normalize_au_values(au_au1_array, max_value=5)
                au_au2_array = normalize_au_values(au_au2_array, max_value=5)
                au_au4_array = normalize_au_values(au_au4_array, max_value=5)
                au_eye_array = normalize_au_values(au_eye_array, max_value=5)
                au_nose_array = normalize_au_values(au_nose_array, max_value=5)
                
                au_features_custom = {
                    "au1": au_au1_array,
                    "au2": au_au2_array,
                    "au4": au_au4_array,
                    "eye": au_eye_array,
                    "nose": au_nose_array,
                }
                print(f"[INFO] 成功从 CSV 读取 AU 值（自动归一化到 0-1）: au1={au_au1_array.shape}, au2={au_au2_array.shape}, au4={au_au4_array.shape}, eye={au_eye_array.shape}, nose={au_nose_array.shape}")
                 
        except Exception as e:
            print(f"[WARN] 从 CSV 读取 AU 值失败: {e}")
            print(f"[WARN] 回退到从视频提取 AU 值...")
            
            if video_path_for_fallback is None:
                raise FileNotFoundError(f"在 {path} 中未找到 .mp4 视频文件（回退提取 AU 需要）")
            
            # 回退：从视频提取
            jaanet_checkpoints_dir = getattr(args, "jaanet_checkpoints_dir", None) if args else None
            jaanet_dataset = getattr(args, "jaanet_dataset", "DISFA") if args else "DISFA"
            jaanet_device = getattr(args, "jaanet_device", "cuda") if args else "cuda"
            jaanet_version = getattr(args, "jaanet_version", "v1") if args else "v1"
            
            print(f"[INFO] 从视频提取自定义 AU（使用 JAANet）: {video_path_for_fallback}")
            au_features_custom = extract_au_from_video_jaanet(
                video_path=video_path_for_fallback,
                num_frames=custom_num_frames,
                jaanet_checkpoints_dir=jaanet_checkpoints_dir,
                jaanet_dataset=jaanet_dataset,
                jaanet_device=jaanet_device,
                jaanet_version=jaanet_version,
            )
            #0312结束
 #0130结束 
    

        current_len = len(eye_features_custom)
        if current_len > target_frame_count:
            print(f"[INFO] Custom AU length ({current_len}) > frames ({target_frame_count}), truncating...")
            eye_features_custom = eye_features_custom[:target_frame_count]
            au_features_custom = {k: v[:target_frame_count] for k, v in au_features_custom.items()}
        elif current_len < target_frame_count:
            pad_len = target_frame_count - current_len
            print(f"[INFO] Custom AU length ({current_len}) < frames ({target_frame_count}), padding {pad_len} zeros...")
            eye_features_custom = np.pad(eye_features_custom, (0, pad_len), 'constant', constant_values=0)
            au_features_custom = {
                k: np.pad(v, ((0, pad_len), (0, 0)), mode="constant")
                for k, v in au_features_custom.items()
            }
        else:
            print(f"[INFO] Custom AU length ({current_len}) matches frames ({target_frame_count})")

        print(f"[INFO] len(frames) = {target_frame_count}")
        print(f"[INFO] len(custom_au_csv) = {len(eye_features_custom)}")
        use_custom_au = True

    if custom_aud:
        aud_features = np.load(os.path.join(path, custom_aud))
        aud_features = torch.from_numpy(aud_features)
        if len(aud_features.shape) == 3:
            aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16]    
        else:
            raise NotImplementedError(f'[ERROR] aud_features.shape {aud_features.shape} not supported')
        print("Reading Custom Transforms")
        eye_features_to_use = eye_features_custom if use_custom_au else eye_features
        au_features_to_use = au_features_custom if use_custom_au else au_features
        custom_cam_infos = readCamerasFromTracksTransforms(path, "track_params.pt", "transforms_val.json", aud_features, eye_features_to_use, au_features_to_use, extension, 
                                                           timestamp_mapper,custom_aud=custom_aud, use_custom_au=use_custom_au, args=args)
    else:
        custom_cam_infos=None
        
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    


    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 2000
        # print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        
        # Initialize with 3DMM Vertices
        facial_mesh = torch.load(mesh_path)["vertices"]
        average_facial_mesh = torch.mean(facial_mesh, dim=0)
        xyz = average_facial_mesh.cpu().numpy()
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        
        
        
    else:
        raise NotImplementedError("No ply file found!")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           custom_cameras=custom_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info


sceneLoadTypeCallbacks2 = {
    "ER-NeRF": readTalkingPortraitDatasetInfo,
}
