import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from utils.mask_utils import generate_3d_point_masks_from_coords
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
from scene.networks import AudioNet, AudioAttNet, MLP
from scene.canonical_tri_plane import canonical_tri_plane
from scene.transformer.transformer import Spatial_Audio_Attention_Module

# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.grid_pe = grid_pe
        
        # audio network
        self.audio_in_dim = 29    
        self.audio_dim = 32
        self.eye_dim = 1
        self.au_au1_dim = 1   # AU1（独立处理）
        self.au_au2_dim = 1   # AU2（独立处理）
        self.au_au4_dim = 1   # AU4（独立处理）
        self.au_eye_dim = 1   # 眼部组 AU6（从JAANet提取，缺少AU5和AU7）
        self.au_nose_dim = 1  # 鼻子组 AU9（从JAANet提取）
        
        self.no_grid = args.no_grid
        self.tri_plane = canonical_tri_plane(args,self.D, self.W)
        
        self.args = args
        self.only_infer = args.only_infer
        self.enc_x = None
        self.aud_ch_att = None
        
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.transformer = Spatial_Audio_Attention_Module(args)
        self.ratio=0
        self.create_net()
        
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)
        self.audio_mlp = MLP(32, args.d_model, 64, 2)
        
        self.in_dim = 32 
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
        
        
        self.eye_mlp = MLP(32, args.d_model, 64, 2)
        self.au_au1_mlp = MLP(32, args.d_model, 64, 2)  # AU1投影：32维 -> d_model维
        self.au_au2_mlp = MLP(32, args.d_model, 64, 2)  # AU2投影：32维 -> d_model维
        self.au_au4_mlp = MLP(32, args.d_model, 64, 2)  # AU4投影：32维 -> d_model维
        self.au_eye_mlp = MLP(32, args.d_model, 64, 2)   # 眼部组投影：32维 -> d_model维
        self.au_nose_mlp = MLP(32, args.d_model, 64, 2)  # 鼻子组投影：32维 -> d_model维
        # AU编码器：为每个AU创建独立的编码器，确保它们能够学习到不同的特征表示
        self.au_au1_encoder = MLP(1, 32, 64, 2)  # AU1编码器：输入1维，输出32维
        self.au_au2_encoder = MLP(1, 32, 64, 2)  # AU2编码器：输入1维，输出32维
        self.au_au4_encoder = MLP(1, 32, 64, 2)  # AU4编码器：输入1维，输出32维
        self.au_eye_encoder = MLP(1, 32, 64, 2)  # 眼部组编码器：输入1维，输出32维
        self.au_nose_encoder = MLP(1, 32, 64, 2)  # 鼻子组编码器：输入1维，输出32维
        # 保留通用编码器用于向后兼容（如果遇到未知类型的AU）
        self.au_mlp_encoder = MLP(1, 32, 64, 2)  # 通用AU编码器（向后兼容）
        self.cam_mlp = MLP(12, args.d_model, 64, 2)
        self.null_vector = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.enc_x_mlp = MLP(32, args.d_model, 64, 2)
        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.tri_plane.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.args.d_model,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))
        
    
    def mlp_init_zeros(self):
        nn.init.xavier_uniform_(self.pos_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.pos_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.scales_deform[-1].weight,gain=0.1) 
        nn.init.zeros_(self.scales_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.rotations_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.rotations_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.opacity_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.opacity_deform[-1].bias)
        
        nn.init.xavier_uniform_(self.shs_deform[-1].weight,gain=0.1)
        nn.init.zeros_(self.shs_deform[-1].bias)
        
        #triplane
        self.tri_plane.mlp_init_zeros()
    def mlp2cpu(self):
        self.tri_plane.mlp2cpu()
        
    def eye_encoding(self,value, d_model=32):
        batch_size, _ = value.shape  
        value = value.to('cuda')

        positions = torch.arange(d_model, device='cuda').float()
        div_term = torch.pow(10000, (2 * (positions // 2)) / d_model)

        encoded_vec = torch.zeros(batch_size, d_model, device='cuda')
        encoded_vec[:, 0::2] = torch.sin(value * div_term[::2])  
        encoded_vec[:, 1::2] = torch.cos(value * div_term[1::2])

        return encoded_vec
    
    def au_encoding(self, value, d_model=32, batch_size=None, au_type=None):
        """
        通用AU编码：支持标量或向量AU；None时返回全零，占位不触发变形。
        
        使用可学习的MLP编码器替代sin/cos编码，更适合非周期性的肌肉变化。
        
        参数:
            value: AU值，形状为 [B, 1] 或 None
            d_model: 输出维度（默认32）
            batch_size: 批次大小（当value为None时使用）
            au_type: AU类型，可选值："au1", "au2", "au4", "eye", "nose"，如果为None则使用通用编码器
        """
        if value is None:
            if batch_size is None:
                batch_size = 1
            return torch.zeros(batch_size, d_model, device="cuda")

        batch_size = value.shape[0]
        # 处理多维AU（已废弃，现在每个AU独立处理）
        if value.shape[1] > 1:
            value = value[:, 0:1]  # 只取第一个
        
        # 根据AU类型选择对应的编码器
        if au_type == "au1":
            return self.au_au1_encoder(value)  # [B, 1] -> [B, 32]
        elif au_type == "au2":
            return self.au_au2_encoder(value)  # [B, 1] -> [B, 32]
        elif au_type == "au4":
            return self.au_au4_encoder(value)  # [B, 1] -> [B, 32]
        elif au_type == "eye":
            return self.au_eye_encoder(value)  # [B, 1] -> [B, 32]
        elif au_type == "nose":
            return self.au_nose_encoder(value)  # [B, 1] -> [B, 32]
        else:
            # 向后兼容：如果未指定类型，使用通用编码器
            return self.au_mlp_encoder(value)  # [B, 1] -> [B, 32]
    
    def query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features):
        # audio_features [1, 8, 29, 16]) 
        audio_features = audio_features.squeeze(0)
        enc_a = self.audio_net(audio_features)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        
        if self.only_infer:
            if self.enc_x == None:
                self.enc_x = self.grid(rays_pts_emb[:,:3]) 
                self.aud_ch_att = self.aud_ch_att_net(self.enc_x)
            enc_x = self.enc_x
            aud_ch_att = self.aud_ch_att
            
        else:
            enc_x = self.grid(rays_pts_emb[:,:3])   # shape ; # , 32
            aud_ch_att = self.aud_ch_att_net(enc_x)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        
        enc_w = enc_a * aud_ch_att
        eye_att = self.eye_att_net(enc_x)
        eye_att = torch.sigmoid(eye_att)
        eye_features = eye_features * eye_att
        hidden = torch.cat([enc_x, enc_w, eye_features], dim=-1)            # # , 65(32+32+1)
    
        hidden = self.feature_out(hidden)   
  
        return hidden
    
    def attention_query_audio_batch(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features, cam_features, au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        # audio_features [B, 8, 29, 16]) 
        B, _, _, _= audio_features.shape
        
        if self.only_infer: #cashing
            if self.enc_x == None:
                self.enc_x = self.tri_plane(rays_pts_emb,only_feature = True, train_tri_plane = self.args.train_tri_plane)
            enc_x = self.enc_x[:B]
        
        else:
            # audio_features [B, 8, 29, 16]) 
            enc_x = self.tri_plane(rays_pts_emb,only_feature = True, train_tri_plane = self.args.train_tri_plane)
            
        enc_a_list= []
        for i in range(B):
            enc_a = self.audio_net(audio_features[i])   # 8 32
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # 1 32 
            enc_a_list.append(enc_a.unsqueeze(0))
        
        enc_a = torch.cat(enc_a_list,dim=0) # B, 1, 32
        enc_eye = self.eye_encoding(eye_features).unsqueeze(1) # B, 1, 32
        #2.1修改 - 每个AU独立处理，使用独立的编码器
        enc_au1 = self.au_encoding(au_au1_features, batch_size=B, au_type="au1").unsqueeze(1)  # (B, 1, 32)
        enc_au2 = self.au_encoding(au_au2_features, batch_size=B, au_type="au2").unsqueeze(1)  # (B, 1, 32)
        enc_au4 = self.au_encoding(au_au4_features, batch_size=B, au_type="au4").unsqueeze(1)  # (B, 1, 32)
        enc_eye_group = self.au_encoding(au_eye_features, batch_size=B, au_type="eye").unsqueeze(1)  # (B, 1, 32)
        enc_nose = self.au_encoding(au_nose_features, batch_size=B, au_type="nose").unsqueeze(1)  # (B, 1, 32)
        enc_cam = self.cam_mlp(cam_features).unsqueeze(1)
        
        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a) # B, 1, dim
            enc_eye = self.eye_mlp(enc_eye) # B, 1, dim
            enc_au1 = self.au_au1_mlp(enc_au1)
            enc_au2 = self.au_au2_mlp(enc_au2)
            enc_au4 = self.au_au4_mlp(enc_au4)
            enc_eye_group = self.au_eye_mlp(enc_eye_group)
            enc_nose = self.au_nose_mlp(enc_nose)
        
        enc_source = torch.cat([enc_a,enc_eye, 
            enc_au1,  # AU1
            enc_au2,  # AU2
            enc_au4,  # AU4
            enc_eye_group,  # 眼部组
            enc_nose,  # 鼻子组
            enc_cam, self.null_vector.repeat(B,1,1)],dim = 1) # B, 8, dim
        x, attention = self.transformer(enc_x, enc_source)

        # 1.26 新添加：计算区域注意力统计值，用于注意力约束 Loss
        if getattr(self.args, 'use_attention_constraint', False):
            # 获取 3D 坐标并生成 mask (基于3D 坐标阈值)
            # rays_pts_emb 前 3 维是原始坐标
            points_3d = rays_pts_emb[:, :3] if rays_pts_emb.dim() == 2 else rays_pts_emb[:, :, :3]
            upper_mask, mouth_mask = generate_3d_point_masks_from_coords(points_3d, method='quantile')
            
            # 取最后一层注意力 (B, N, C)
            att_last = attention[:, -1] if attention.dim() == 4 else attention
            
            # 初始化统计值
            self.audio_attn_upper = torch.tensor(0.0, device=x.device)
            self.au_attn_upper = torch.tensor(0.0, device=x.device)
            self.audio_attn_mouth = torch.tensor(0.0, device=x.device)
            self.au_attn_mouth = torch.tensor(0.0, device=x.device)
            
            # 分区域统计
            audio_attn_upper_list = []
            au_attn_upper_list = []
            audio_attn_mouth_list = []
            au_attn_mouth_list = []
            
            for b in range(B):
                att_b = att_last[b]
                # 上半脸
                if upper_mask.sum() > 0:
                    attn_upper_b = att_b[upper_mask]
                    audio_attn_upper_list.append(attn_upper_b[:, 0].mean()) # token 0: 音频
                    au_attn_upper_list.append(attn_upper_b[:, 2:7].sum(dim=-1).mean()) # token 2-5: AU1, AU2, AU4, AU6(眼部组)，AU9
                # 嘴部
                if mouth_mask.sum() > 0:
                    attn_mouth_b = att_b[mouth_mask]
                    audio_attn_mouth_list.append(attn_mouth_b[:, 0].mean())
                    au_attn_mouth_list.append(attn_mouth_b[:, 2:7].sum(dim=-1).mean()) # token 2-5: AU1, AU2, AU4, AU6，AU9
            
            if audio_attn_upper_list:
                self.audio_attn_upper = torch.stack(audio_attn_upper_list).mean()
                self.au_attn_upper = torch.stack(au_attn_upper_list).mean()
            if audio_attn_mouth_list:
                self.audio_attn_mouth = torch.stack(audio_attn_mouth_list).mean()
                self.au_attn_mouth = torch.stack(au_attn_mouth_list).mean()
        #1.26新添加结束
        #1.26新添加调试信息
        # 注意：transformer 返回的 attention 含 layer 维，形状通常为 (B, n_layer, N, num_conditions)
        # 其中 num_conditions=9: 音频、AU45、AU1、AU2、AU4、眼部组、鼻子组、相机、null
        if getattr(self.args, 'debug_attn', False):
            condition_names = ['音频', 'AU45(眨眼)', 'AU1', 'AU2', 'AU4', '眼部组', '鼻子组', '相机', 'null']

            # 兼容两种可能的 shape：
            # - (B, n_layer, N, C)  : 本项目 transformer 的默认返回
            # - (B, N, C)           : 若后续有人改动 transformer
            if attention.dim() == 4:
                # 只看最后一层注意力（更直观，也更常用）
                att_last = attention[:, -1]  # (B, N, C)
            else:
                att_last = attention  # (B, N, C)

            # 对 batch 和点取平均，得到每个 token 的平均注意力 (C,)
            attn_mean = att_last.mean(dim=(0, 1))  # (C,)
            attn_mean_cpu = attn_mean.detach().float().cpu().numpy()

            print(f"\n[调试] 注意力权重分布（批次版本，最后一层，共{len(condition_names)}个条件）：")
            total_sum = 0.0
            for i, name in enumerate(condition_names):
                w = float(attn_mean_cpu[i])
                total_sum += w
                print(f"  {i}. {name}: {w:.4f} ({w*100:.2f}%)")

            # sanity check：随机取一个点，检查该点对所有 token 的注意力是否近似归一化为 1
            b0 = 0
            n0 = int(np.random.randint(0, att_last.shape[1])) if att_last.shape[1] > 0 else 0
            one_point_sum = float(att_last[b0, n0, :].sum().detach().cpu())
            print(f"  总计(均值): {total_sum:.4f} | 单点sum(b={b0}, n={n0}): {one_point_sum:.4f}\n")
        #1.26新添加修改结束
        
        return x, attention
    
    def attention_query_audio(self, rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features, au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        # audio_features [1, 8, 29, 16])
        if self.only_infer:
            if self.enc_x == None:
                enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0),only_feature = True, train_tri_plane = self.args.train_tri_plane).unsqueeze(0)
            enc_x = self.enc_x
            
        else:
            enc_x = self.tri_plane(rays_pts_emb.unsqueeze(0),only_feature = True, train_tri_plane = self.args.train_tri_plane).unsqueeze(0)
        
        audio_features = audio_features.squeeze(0) # 8, 29, 16
        enc_a = self.audio_net(audio_features)  # 8, 32
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)).unsqueeze(0) # 1, 1, 32
        
        enc_eye = self.eye_encoding(eye_features).unsqueeze(1) # 1, 1, 32
        #2.1修改 - 每个AU独立处理
        enc_au1 = self.au_encoding(au_au1_features, batch_size=1, au_type="au1").unsqueeze(1)  # (1, 1, 32)
        enc_au2 = self.au_encoding(au_au2_features, batch_size=1, au_type="au2").unsqueeze(1)  # (1, 1, 32)
        enc_au4 = self.au_encoding(au_au4_features, batch_size=1, au_type="au4").unsqueeze(1)  # (1, 1, 32)
        enc_eye_group = self.au_encoding(au_eye_features, batch_size=1, au_type="eye").unsqueeze(1)  # (1, 1, 32)
        enc_nose = self.au_encoding(au_nose_features, batch_size=1, au_type="nose").unsqueeze(1)  # (1, 1, 32)
        #1_22新添加结束
        
        if self.args.d_model != 32:
            enc_a = self.audio_mlp(enc_a)
            enc_eye = self.eye_mlp(enc_eye)
            #2.1新添加 - 每个AU独立投影
            enc_au1 = self.au_au1_mlp(enc_au1)
            enc_au2 = self.au_au2_mlp(enc_au2)
            enc_au4 = self.au_au4_mlp(enc_au4)
            enc_eye_group = self.au_eye_mlp(enc_eye_group)
            enc_nose = self.au_nose_mlp(enc_nose)            
            #1_22新添加结束

        enc_source = torch.cat([enc_a,enc_eye,enc_au1,enc_au2,enc_au4,enc_eye_group, enc_nose, self.null_vector],dim = 1) # 1, 8, dim
        #1_22新添加结束
        
        x, attention = self.transformer(enc_x, enc_source) # 1, N, dim
        #调试输出：打印注意力权重
        #1.26新添加修改
        # 注意：transformer 返回的 attention 含 layer 维，形状通常为 (B, n_layer, N, num_conditions)
        # 这里 num_conditions=8: 音频、AU45、AU1、AU2、AU4、眼部组、鼻子组、null（单批次版本无相机 token）
        if getattr(self.args, 'debug_attn', False):
            condition_names = ['音频', 'AU45(眨眼)', 'AU1', 'AU2', 'AU4', '眼部组', '鼻子组', 'null']

            if attention.dim() == 4:
                att_last = attention[:, -1]  # (B, N, C)
            else:
                att_last = attention  # (B, N, C)

            attn_mean = att_last.mean(dim=(0, 1))  # (C,)
            attn_mean_cpu = attn_mean.detach().float().cpu().numpy()

            print(f"\n[调试] 注意力权重分布（单批次版本，最后一层，共{len(condition_names)}个条件）：")
            total_sum = 0.0
            for i, name in enumerate(condition_names):
                w = float(attn_mean_cpu[i])
                total_sum += w
                print(f"  {i}. {name}: {w:.4f} ({w*100:.2f}%)")

            b0 = 0
            n0 = int(np.random.randint(0, att_last.shape[1])) if att_last.shape[1] > 0 else 0
            one_point_sum = float(att_last[b0, n0, :].sum().detach().cpu())
            print(f"  总计(均值): {total_sum:.4f} | 单点sum(b={b0}, n={n0}): {one_point_sum:.4f}\n")
        #1.26新添加修改结束

        x = x.squeeze()
        return x, attention

    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, shs_emb=None,
                audio_features=None, eye_features=None, cam_features=None,
                au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        if audio_features is None:
            return self.forward_static(rays_pts_emb)
        elif len(rays_pts_emb.shape)==3:
            return self.forward_dynamic_batch(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, 
                audio_features, eye_features, cam_features,
                au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features)
        elif len(rays_pts_emb.shape)==2:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, 
                audio_features, eye_features,
                au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features)

    def forward_static(self, rays_pts_emb):
        grid_feature, scale, rotation, opacity, sh = self.tri_plane(rays_pts_emb)
        # dx = self.static_mlp(grid_feature)
        return rays_pts_emb, scale, rotation, opacity, sh.reshape([sh.shape[0],sh.shape[1],16,3])
    
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features, eye_features,
                        au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        hidden, attention = self.attention_query_audio(
            rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features,
            au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features
            )
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
            
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb*mask.unsqueeze(-1) + dshs
            
        return pts, scales, rotations, opacity, shs, attention
    
    def forward_dynamic_batch(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, audio_features, eye_features, cam_features,
                               au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        hidden, attention = self.attention_query_audio_batch(
            rays_pts_emb, scales_emb, rotations_emb, audio_features, eye_features, cam_features,
            au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features
            )
        B, _, _, _ = audio_features.shape
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,:,0]).unsqueeze(-1)
        if self.args.no_dx:
            pts = rays_pts_emb[:,:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:,:3])
            pts = rays_pts_emb[:B,:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:,:3])
            scales = scales_emb[:B,:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:B,:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:,:1])
            opacity = opacity_emb[:B,:,:1]*mask + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],shs_emb.shape[1],16,3])

            shs = torch.zeros_like(shs_emb)
            shs = shs_emb[:B]*mask.unsqueeze(-1) + dshs
        
        return pts, scales, rotations, opacity, shs, attention
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, args=args)
        self.only_infer = args.only_infer
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        self.deformation_net.mlp_init_zeros()
        
        self.point_emb = None
        self.scales_emb = None
        self.rotations_emb = None
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, 
                audio_features = None, eye_features=None, cam_features =None,
                au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        if audio_features is not None:
            return self.forward_dynamic(
                point, scales, rotations, opacity, shs, 
                audio_features, eye_features, cam_features,
                au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features)
        else:
            return self.forward_static(point, scales, rotations, opacity, shs)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points, scales, rotations, opacity, shs):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, 
                        audio_features=None, eye_features=None, cam_features=None,
                        au_au1_features=None, au_au2_features=None, au_au4_features=None, au_eye_features=None, au_nose_features=None):
        if self.only_infer:
            if self.point_emb == None:
                self.point_emb = poc_fre(point,self.pos_poc)         # #, 3 -> #, 63
                self.scales_emb = poc_fre(scales,self.rotation_scaling_poc)
                self.rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
                
            point_emb = self.point_emb
            scales_emb = self.scales_emb
            rotations_emb = self.rotations_emb

            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                    scales_emb,
                                                    rotations_emb,
                                                    opacity,
                                                    shs,
                                                    audio_features, eye_features, cam_features,
                                                    au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features)
            return means3D, scales, rotations, opacity, shs, attention

        else:
            point_emb = poc_fre(point,self.pos_poc)         # B, N, 3 -> B, N, 63
            scales_emb = poc_fre(scales,self.rotation_scaling_poc)
            rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
            means3D, scales, rotations, opacity, shs, attention = self.deformation_net(point_emb,
                                                    scales_emb,
                                                    rotations_emb,
                                                    opacity,
                                                    shs,
                                                    audio_features, eye_features, cam_features,
                                                    au_au1_features, au_au2_features, au_au4_features, au_eye_features, au_nose_features)
            return means3D, scales, rotations, opacity, shs, attention
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    def mlp2cpu(self):
        self.deformation_net.mlp2cpu()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
def poc_fre(input_data,poc_buf):
    
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb