"""
JAANet AU 提取器

封装 PyTorch-JAANet（PyTorch-JAANet/）项目，用于从图像张量中提取 AU 概率。
主要用于训练脚本中的 AU loss（teacher 模型），接口风格与
`utils/graphau_extractor.py`、`utils/opengraphau_extractor.py` 保持一致。
"""

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ----------------------------- 定位 PyTorch-JAANet 路径 ----------------------------- #

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
JAANET_PATH = os.path.join(project_root, "PyTorch-JAANet")

if not os.path.exists(JAANET_PATH):
    raise FileNotFoundError(f"PyTorch-JAANet not found at: {JAANET_PATH}")

if JAANET_PATH not in sys.path:
    sys.path.insert(0, JAANET_PATH)

import network  # type: ignore


JAANET_BP4D_AU_LIST: List[str] = [
    "AU1",
    "AU2",
    "AU4",
    "AU6",
    "AU7",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU23",
    "AU24",
]

JAANET_DISFA_AU_LIST: List[str] = [
    "AU1",
    "AU2",
    "AU4",
    "AU6",
    "AU9",
    "AU12",
    "AU25",
    "AU26",
]


def _resolve_ckpt_path(checkpoints_dir: str, name: str, epoch: Optional[int]) -> str:
    """
    根据模块名和 epoch 解析 checkpoint 路径。

    优先尝试 name_{epoch}.pth，其次尝试 name.pth。
    """
    if epoch is not None:
        filename_epoch = f"{name}_{epoch}.pth"
        path_epoch = os.path.join(checkpoints_dir, filename_epoch)
        if os.path.exists(path_epoch):
            return path_epoch

    filename_plain = f"{name}.pth"
    path_plain = os.path.join(checkpoints_dir, filename_plain)
    if os.path.exists(path_plain):
        return path_plain

    if epoch is not None:
        tried = f"{filename_epoch} 或 {filename_plain}"
    else:
        tried = filename_plain
    raise FileNotFoundError(
        f"在 checkpoints_dir={checkpoints_dir} 中未找到权重文件（期望 {tried}）。"
    )


def build_jaanet_modules(
    checkpoints_dir: str,
    au_num: int,
    crop_size: int = 176,
    map_size: int = 44,
    land_num: int = 49,
    unit_dim: int = 8,
    device: str = "cuda",
    version: str = "v1",
    epoch: Optional[int] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    """
    构建并加载 JAANet 的 6 个子模块：
    - HMRegionLearning
    - AlignNet
    - LocalAttentionRefine
    - LocalAUNetv1/v2
    - HLFeatExtractor（global_au_feat）
    - AUNet
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"checkpoints_dir 不存在: {checkpoints_dir}")

    region_learning = network.network_dict["HMRegionLearning"](
        input_dim=3, unit_dim=unit_dim
    )
    align_net = network.network_dict["AlignNet"](
        crop_size=crop_size,
        map_size=map_size,
        au_num=au_num,
        land_num=land_num,
        input_dim=unit_dim * 8,
    )
    local_attention_refine = network.network_dict["LocalAttentionRefine"](
        au_num=au_num, unit_dim=unit_dim
    )
    if version.lower() == "v1":
        local_au_key = "LocalAUNetv1"
    elif version.lower() == "v2":
        local_au_key = "LocalAUNetv2"
    else:
        raise ValueError(f"不支持的 JAANet 版本: {version}，应为 v1 或 v2。")

    local_au_net = network.network_dict[local_au_key](
        au_num=au_num, input_dim=unit_dim * 8, unit_dim=unit_dim
    )
    global_au_feat = network.network_dict["HLFeatExtractor"](
        input_dim=unit_dim * 8, unit_dim=unit_dim
    )
    au_net = network.network_dict["AUNet"](
        au_num=au_num, input_dim=12000, unit_dim=unit_dim
    )

    module_names = [
        ("region_learning", region_learning),
        ("align_net", align_net),
        ("local_attention_refine", local_attention_refine),
        ("local_au_net", local_au_net),
        ("global_au_feat", global_au_feat),
        ("au_net", au_net),
    ]
    map_location = torch.device(device if torch.cuda.is_available() else "cpu")
    for name, module in module_names:
        ckpt_path = _resolve_ckpt_path(checkpoints_dir, name, epoch)
        state = torch.load(ckpt_path, map_location=map_location)
        try:
            module.load_state_dict(state, strict=True)
            print(f"[INFO] [JAANet] 加载权重: {ckpt_path}")
        except RuntimeError as e:
            # 如果严格加载失败，尝试部分加载（允许缺失的键）
            print(f"[WARN] [JAANet] 严格加载权重失败: {ckpt_path}")
            print(f"[WARN] 错误信息: {str(e)[:200]}...")
            print(f"[WARN] 尝试部分加载（忽略缺失的键）...")
            try:
                module.load_state_dict(state, strict=False)
                print(f"[INFO] [JAANet] 部分加载权重成功: {ckpt_path}")
            except Exception as e2:
                # 分析错误信息，尝试推断权重文件是为多少个AU训练的
                error_str = str(e)
                missing_keys = []
                if "Missing key(s)" in error_str:
                    # 尝试从错误信息中提取缺失的键
                    import re
                    missing_pattern = r'"local_aus_attention\.(\d+)\.'
                    missing_indices = re.findall(missing_pattern, error_str)
                    if missing_indices:
                        max_index = max([int(idx) for idx in missing_indices])
                        inferred_au_num = max_index + 1  # 索引从0开始
                        print(f"[INFO] 根据错误信息推断，权重文件可能是为 {inferred_au_num} 个 AU 训练的")
                
                raise RuntimeError(
                    f"无法加载权重文件 {ckpt_path}。\n"
                    f"可能原因：权重文件与模型配置不匹配（AU数量不一致）。\n"
                    f"当前配置：au_num={au_num}（{'BP4D' if au_num == 12 else 'DISFA' if au_num == 8 else '未知'}），\n"
                    f"请确保权重文件是为相同的 AU 数量训练的，或检查 --jaanet_dataset 参数是否正确。\n"
                    f"原始错误: {str(e)[:500]}"
                ) from e2

    for _, module in module_names:
        module.to(device)
        module.eval()

    return (
        region_learning,
        align_net,
        local_attention_refine,
        local_au_net,
        global_au_feat,
        au_net,
    )


class JaaNetExtractor(nn.Module):
    """
    使用 PyTorch-JAANet 从图像张量中提取 AU 概率。

    输入：
        images: [B, 3, H, W]，值域 [0, 1]，RGB
    输出：
        [B, K]，K 为 AU 数量（BP4D=12，DISFA=8），为 AU 发生概率（0~1）。
    """

    def __init__(
        self,
        checkpoints_dir: str,
        dataset: str = "DISFA",
        device: str = "cuda",
        version: str = "v1",
        epoch: Optional[int] = None,
        crop_size: int = 176,
        map_size: int = 44,
        land_num: int = 49,
        unit_dim: int = 8,
    ):
        super().__init__()
        self.device = device
        self.version = version.lower()
        self.crop_size = crop_size

        ds = dataset.upper()
        if ds == "BP4D":
            self.au_names = JAANET_BP4D_AU_LIST
        elif ds == "DISFA":
            self.au_names = JAANET_DISFA_AU_LIST
        else:
            raise ValueError(f"不支持的数据集类型: {dataset}（应为 BP4D 或 DISFA）")
        au_num = len(self.au_names)

        (
            self.region_learning,
            self.align_net,
            self.local_attention_refine,
            self.local_au_net,
            self.global_au_feat,
            self.au_net,
        ) = build_jaanet_modules(
            checkpoints_dir=checkpoints_dir,
            au_num=au_num,
            crop_size=crop_size,
            map_size=map_size,
            land_num=land_num,
            unit_dim=unit_dim,
            device=device,
            version=version,
            epoch=epoch,
        )

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        预处理到 JAANet 期望的形状与归一化（与官方预处理一致）。
        
        处理流程（与 PyTorch-JAANet/pre_process.py 的 image_test 一致）：
        1. 如果图像小于 crop_size，先等比例放大到至少 crop_size（安全处理）
        2. CenterCrop 到 [crop_size, crop_size]
        3. Normalize 到 [-1, 1]（mean=0.5, std=0.5）
        
        注意：官方预处理直接 CenterCrop，但为了处理小图像，这里先检查大小。
        如果输入图像已经 >= crop_size，则行为与官方预处理完全一致。
        """
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        elif len(images.shape) == 5:
            images = images[:, 0]
        elif len(images.shape) != 4:
            raise ValueError(f"不支持的图像形状: {images.shape}，期望 [B, 3, H, W]")

        images = images.to(self.device)

        _, _, h, w = images.shape
        target = self.crop_size
        
        # 如果图像小于 crop_size，先等比例放大（确保 CenterCrop 能正常工作）
        # 这与官方预处理略有不同，但更安全。如果输入已经 >= crop_size，则行为一致。
        if h < target or w < target:
            # 等比例放大到 min(H, W) >= target
            scale = max(target / h, target / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            images = torch.nn.functional.interpolate(
                images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            h, w = new_h, new_w

        # CenterCrop 到 [crop_size, crop_size]（与官方预处理一致）
        top = (h - target) // 2
        left = (w - target) // 2
        images = images[:, :, top : top + target, left : left + target]

        # Normalize 到 [-1, 1]（mean=0.5, std=0.5，与官方预处理一致）
        mean = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：返回 AU 概率 [B, K]。
        """
        x = self._preprocess(images)  # [B, 3, crop_size, crop_size]

        with torch.no_grad():
            region_feat = self.region_learning(x)
            align_feat, align_output, aus_map = self.align_net(region_feat)
            aus_map = aus_map.to(self.device)

            output_aus_map = self.local_attention_refine(aus_map.detach())

            if self.version == "v1":
                local_au_out_feat = self.local_au_net(region_feat, output_aus_map)
            else:
                local_au_out_feat, _ = self.local_au_net(region_feat, output_aus_map)

            global_au_out_feat = self.global_au_feat(region_feat)
            concat_au_feat = torch.cat(
                (align_feat, global_au_out_feat, local_au_out_feat), 1
            )
            aus_output = self.au_net(concat_au_feat)

            # log_softmax -> 概率，取“发生”这一类（索引 1）
            prob = (aus_output[:, 1, :]).exp()

        return prob

