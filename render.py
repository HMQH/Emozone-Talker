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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_from_batch
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import concurrent.futures
import shutil
import subprocess
from torch.utils.data import DataLoader


def mux_audio(video_path, audio_path, output_path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", video_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
        ],
        check=True,
    )

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, scene, gaussians, pipeline,
               audio_dir, batch_size, num_workers):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir
    
    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)
    
    viewpoint_stack = scene
    viewpoint_stack_loader = DataLoader(
        viewpoint_stack,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list,
    )
    
    loader = iter(viewpoint_stack_loader)
    
    if name == "train" :
        process_until = 1000
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        process_until = len(viewpoint_stack.dataset) 
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------") 
    print("point nums:",gaussians._xyz.shape[0])
    image = []
    gt = []
    audio_attention = []
    eye_attention = []
    null_attention = []
    cam_attention = []
    #1.26新添加修改
    mask_upper_attention = []
    mask_mouth_attention = []
    #1.26新添加修改结束
    iterations = process_until // batch_size
    if process_until % batch_size != 0:
        iterations += 1
    total_time = 0
    #render image
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):

        viewpoint_cams = next(loader)
        # try:
        #     output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
        #                         random_color= False, stage='fine',
        #                         batch_size=batch_size, visualize_attention=False, only_infer=True)
        # except:
        #     break
        # 新添加！！！！！！！！！！！！！！！！！！！！
        try:
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                random_color= False, stage='fine',
                                batch_size=batch_size, visualize_attention=False, only_infer=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise       

        total_time += output["inference_time"]
        image.append(output["rendered_image_tensor"].cpu())
        gt.append(output["gt_tensor"].cpu())
        
    image_tensor = torch.cat(image,dim=0)[:process_until]
    gt_image_tensor = torch.cat(gt,dim=0)[:process_until]
    
    print("total frame:",(image_tensor.shape[0]))
    print("FPS:",(torch.cat(image,dim=0).shape[0])/(total_time))
    
    
    #render attention
    loader = iter(viewpoint_stack_loader)
    for idx in range(iterations):

        viewpoint_cams = next(loader)
        # try:
        #     output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
        #                         random_color= False, stage='fine',
        #                         batch_size=batch_size, visualize_attention=True, only_infer=True) 
        # except:
        #     break
        #新添加！！！！！！！！！！！！！！！！
        try:
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                random_color= False, stage='fine',
                                batch_size=batch_size, visualize_attention=True, only_infer=True) 
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

        total_time += output["inference_time"]
        audio_attention.append(output["audio_attention"].cpu())
        eye_attention.append(output["eye_attention"].cpu())
        cam_attention.append(output["cam_attention"].cpu())
        null_attention.append(output["null_attention"].cpu())
        # 1.26 新添加：mask 可视化（需要 gaussian_renderer/__init__.py 返回这两个字段）
        if "mask_upper_attention" in output and output["mask_upper_attention"] is not None:
            mask_upper_attention.append(output["mask_upper_attention"].cpu())
        if "mask_mouth_attention" in output and output["mask_mouth_attention"] is not None:
            mask_mouth_attention.append(output["mask_mouth_attention"].cpu())
        
        
    audio_tensor = torch.cat(audio_attention,0)[:process_until]
    eye_tensor = torch.cat(eye_attention,0)[:process_until]
    cam_tensor = torch.cat(cam_attention,0)[:process_until]
    null_tensor = torch.cat(null_attention,0)[:process_until]
    #1.26新添加：mask可视化
    mask_upper_tensor = torch.cat(mask_upper_attention,0)[:process_until] if len(mask_upper_attention) > 0 else None
    mask_mouth_tensor = torch.cat(mask_mouth_attention,0)[:process_until] if len(mask_mouth_attention) > 0 else None
    
    if name != 'custom':
        write_frames_to_video(tensor_to_image(gt_image_tensor),gts_path+f'/gt', use_imageio = True)
    write_frames_to_video(tensor_to_image(image_tensor),render_path+'/renders', use_imageio = True)
    write_frames_to_video(tensor_to_image(audio_tensor),render_path+'/audio', use_imageio = False)
    write_frames_to_video(tensor_to_image(eye_tensor),render_path+'/eye', use_imageio = False)
    write_frames_to_video(tensor_to_image(null_tensor),render_path+'/null', use_imageio = False)
    write_frames_to_video(tensor_to_image(cam_tensor),render_path+'/cam', use_imageio = False)
    #1.26新添加：mask可视化
    if mask_upper_tensor is not None:
        write_frames_to_video(tensor_to_image(mask_upper_tensor), render_path + '/mask_upper', use_imageio=False)
    if mask_mouth_tensor is not None:
        write_frames_to_video(tensor_to_image(mask_mouth_tensor), render_path + '/mask_mouth', use_imageio=False)
    #1.26新添加结束    
    #0116修改
    model_path_norm = os.path.normpath(model_path)
    model_parts = model_path_norm.split(os.sep)
    model_tag = model_parts[-1]
    
    if name != 'custom':
        mux_audio(
            f"{gts_path}/gt.mp4",
            inf_audio_dir,
            f"{gts_path}/{model_tag}_{name}_{iteration}iter_gt.mp4",
        )
    mux_audio(
        f"{render_path}/renders.mp4", inf_audio_dir,
        f"{render_path}/{model_tag}_{name}_{iteration}iter_renders.mp4",
    )
    mux_audio(
        f"{render_path}/audio.mp4", inf_audio_dir,
        f"{render_path}/{model_tag}_{name}_{iteration}iter_audio.mp4",
    )
    mux_audio(
        f"{render_path}/eye.mp4", inf_audio_dir,
        f"{render_path}/{model_tag}_{name}_{iteration}iter_eye.mp4",
    )
    mux_audio(
        f"{render_path}/null.mp4", inf_audio_dir,
        f"{render_path}/{model_tag}_{name}_{iteration}iter_null.mp4",
    )
    mux_audio(
        f"{render_path}/cam.mp4", inf_audio_dir,
        f"{render_path}/{model_tag}_{name}_{iteration}iter_cam.mp4",
    )
    #1.26新添加：mask可视化
    if mask_upper_tensor is not None:
        mux_audio(
            f"{render_path}/mask_upper.mp4", inf_audio_dir,
            f"{render_path}/{model_tag}_{name}_{iteration}iter_mask_upper.mp4",
        )
    if mask_mouth_tensor is not None:
        mux_audio(
            f"{render_path}/mask_mouth.mp4", inf_audio_dir,
            f"{render_path}/{model_tag}_{name}_{iteration}iter_mask_mouth.mp4",
        )
    #0116修改结束
    if name != 'custom':
        os.remove(f"{gts_path}/gt.mp4")
    os.remove(f"{render_path}/renders.mp4")
    os.remove(f"{render_path}/audio.mp4")
    os.remove(f"{render_path}/eye.mp4")
    os.remove(f"{render_path}/null.mp4")
    os.remove(f"{render_path}/cam.mp4")
    #1.26新添加：mask可视化
    if mask_upper_tensor is not None:
        os.remove(f"{render_path}/mask_upper.mp4")
    if mask_mouth_tensor is not None:
        os.remove(f"{render_path}/mask_mouth.mp4")
    
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train, skip_test, skip_video, batch_size = (
        args.skip_train, args.skip_test, args.skip_video, args.batch
    )
    
    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        #0312修改
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=args.custom_aud, custom_au=args.custom_au, full_args=args)
        
        gaussians.eval()
        
        if args.custom_aud != '':
            audio_dir = os.path.join(data_dir, args.custom_wav)
            render_set(dataset.model_path, "custom", scene.loaded_iter, scene.getCustomCameras(), gaussians, pipeline, audio_dir, batch_size, args.num_workers)
        
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud_train.wav")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size, args.num_workers)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud_novel.wav")
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size, args.num_workers)

def write_frames_to_video(frames, path, codec='mp4v', fps=25, use_imageio=False):
    if use_imageio:
        imageio.mimwrite(f'{path}.mp4', frames, fps=fps, quality=8, output_params=['-vf', f'fps={fps}'], macro_block_size=None)
    else:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(f'{path}.mp4', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()

def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor
        
    if normalize:
        image = 255 * image
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image        

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--batch", type=int, default=8,
                        help="Number of frames rendered per batch")
    parser.add_argument("--num_workers", type=int,
                        default=min(8, os.cpu_count() or 1),
                        help="DataLoader worker processes")
    parser.add_argument("--custom_aud", type=str, default='')
    parser.add_argument("--custom_wav", type=str, default='')
    parser.add_argument("--custom_au", type=str, default='', help="Custom AU CSV file path for inference")
    #0312添加
    parser.add_argument("--use_jaanet_preprocess", action="store_true",
                        help="预处理阶段用 JAANet 从视频提取 AU；否则使用 OpenFace au.csv（默认）")
    # parser.add_argument("--audio_dir", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import runpy
        from utils.params_utils import merge_hparams
        config = runpy.run_path(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    print(args)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)
