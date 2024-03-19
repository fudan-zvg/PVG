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
import glob
import os
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import save_image
from omegaconf import OmegaConf

EPS = 1e-5

@torch.no_grad()
def separation(scene : Scene, renderFunc, renderArgs, env_map=None):
    scale = scene.resolution_scales[0]
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    # we supppose area with altitude>0.5 is static
    # here z axis is downward so is gaussians.get_xyz[:, 2] < -0.5
    high_mask = gaussians.get_xyz[:, 2] < -0.5
    # import pdb;pdb.set_trace()
    mask = (gaussians.get_scaling_t[:, 0] > args.separate_scaling_t) | high_mask
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            outdir = os.path.join(args.model_path, "separation", config['name'])
            os.makedirs(outdir,exist_ok=True)
            for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)
                render_pkg_static = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map, mask=mask)

                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                image_static = torch.clamp(render_pkg_static["render"], 0.0, 1.0)

                save_image(image, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))
                save_image(image_static, os.path.join(outdir, f"{viewpoint.colmap_id:03d}_static.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)

    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'separation')
    os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)
    
    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    separation(scene, render, (args, background), env_map=env_map)

    print("\Rendering statics and dynamics complete.")
