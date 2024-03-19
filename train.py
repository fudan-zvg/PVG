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
import json
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
import numpy as np
import kornia
from omegaconf import OmegaConf
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

EPS = 1e-5
def training(args):
    
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print("Tensorboard not available: not logging progress")
    vis_path = os.path.join(args.model_path, 'visualization')
    os.makedirs(vis_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    
    scene = Scene(args, gaussians)

    gaussians.training_setup(args)
    
    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    first_iter = 0
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)
        
        if env_map is not None:
            env_checkpoint = os.path.join(os.path.dirname(args.checkpoint), 
                                        os.path.basename(args.checkpoint).replace("chkpnt", "env_light_chkpnt"))
            (light_params, _) = torch.load(env_checkpoint)
            env_map.restore(light_params)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, args.iterations + 1), desc="Training progress")
    
    for iteration in progress_bar:       
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = list(range(len(scene.getTrainCameras())))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]
        
        # render v and t scale map
        v = gaussians.get_inst_velocity
        t_scale = gaussians.get_scaling_t.clamp_max(2)
        other = [t_scale, v]

        if np.random.random() < args.lambda_self_supervision:
            time_shift = 3*(np.random.random() - 0.5) * scene.time_interval
        else:
            time_shift = None

        render_pkg = render(viewpoint_cam, gaussians, args, background, env_map=env_map, other=other, time_shift=time_shift, is_training=True)

        image = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha = render_pkg["alpha"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        log_dict = {}

        feature = render_pkg['feature'] / alpha.clamp_min(EPS)
        t_map = feature[0:1]
        v_map = feature[1:]
        
        sky_mask = viewpoint_cam.sky_mask.cuda() if viewpoint_cam.sky_mask is not None else torch.zeros_like(alpha, dtype=torch.bool)

        sky_depth = 900
        depth = depth / alpha.clamp_min(EPS)
        if env_map is not None:
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth
            
        gt_image = viewpoint_cam.original_image.cuda()
        
        loss_l1 = F.l1_loss(image, gt_image)
        log_dict['loss_l1'] = loss_l1.item()
        loss_ssim = 1.0 - ssim(image, gt_image)
        log_dict['loss_ssim'] = loss_ssim.item()
        loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim

        if args.lambda_lidar > 0:
            assert viewpoint_cam.pts_depth is not None
            pts_depth = viewpoint_cam.pts_depth.cuda()

            mask = pts_depth > 0
            loss_lidar =  torch.abs(1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)).mean()
            if args.lidar_decay > 0:
                iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
            else:
                iter_decay = 1
            log_dict['loss_lidar'] = loss_lidar.item()
            loss += iter_decay * args.lambda_lidar * loss_lidar

        if args.lambda_t_reg > 0:
            loss_t_reg = -torch.abs(t_map).mean()
            log_dict['loss_t_reg'] = loss_t_reg.item()
            loss += args.lambda_t_reg * loss_t_reg

        if args.lambda_v_reg > 0:
            loss_v_reg = torch.abs(v_map).mean()
            log_dict['loss_v_reg'] = loss_v_reg.item()
            loss += args.lambda_v_reg * loss_v_reg

        if args.lambda_inv_depth > 0:
            inverse_depth = 1 / (depth + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(inverse_depth[None], gt_image[None])
            log_dict['loss_inv_depth'] = loss_inv_depth.item()
            loss = loss + args.lambda_inv_depth * loss_inv_depth

        if args.lambda_v_smooth > 0:
            loss_v_smooth = kornia.losses.inverse_depth_smoothness_loss(v_map[None], gt_image[None])
            log_dict['loss_v_smooth'] = loss_v_smooth.item()
            loss = loss + args.lambda_v_smooth * loss_v_smooth
        
        if args.lambda_sky_opa > 0:
            o = alpha.clamp(1e-6, 1-1e-6)
            sky = sky_mask.float()
            loss_sky_opa = (-sky * torch.log(1 - o)).mean()
            log_dict['loss_sky_opa'] = loss_sky_opa.item()
            loss = loss + args.lambda_sky_opa * loss_sky_opa

        if args.lambda_opacity_entropy > 0:
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy =  -(o*torch.log(o)).mean()
            log_dict['loss_opacity_entropy'] = loss_opacity_entropy.item()
            loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy
        
        loss.backward()
        log_dict['loss'] = loss.item()
        
        iter_end.record()

        with torch.no_grad():
            psnr_for_log = psnr(image, gt_image).double()
            log_dict["psnr"] = psnr_for_log
            for key in ['loss', "loss_l1", "psnr"]:
                ema_dict_for_log[key] = 0.4 * log_dict[key] + 0.6 * ema_dict_for_log[key]
                
            if iteration % 10 == 0:
                postfix = {k[5:] if k.startswith("loss_") else k:f"{ema_dict_for_log[k]:.{5}f}" for k, v in ema_dict_for_log.items()}
                postfix["scale"] = scene.resolution_scales[scene.scale_index]
                progress_bar.set_postfix(postfix)

            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)
            log_dict['total_points'] = gaussians.get_xyz.shape[0]
            # Log and save
            complete_eval(tb_writer, iteration, args.test_iterations, scene, render, (args, background), 
                          log_dict, env_map=env_map)

            # Densification
            if iteration > args.densify_until_iter * args.time_split_frac:
                gaussians.no_time_split = False

            if iteration < args.densify_until_iter and (args.densify_until_num_points < 0 or gaussians.get_xyz.shape[0] < args.densify_until_num_points):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = args.size_threshold if (iteration > args.opacity_reset_interval and args.prune_big_point > 0) else None

                    if size_threshold is not None:
                        size_threshold = size_threshold // scene.resolution_scales[0]

                    gaussians.densify_and_prune(args.densify_grad_threshold, args.thresh_opa_prune, scene.cameras_extent, size_threshold, args.densify_grad_t_threshold)

                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if env_map is not None and iteration < args.env_optimize_until:
                env_map.optimizer.step()
                env_map.optimizer.zero_grad(set_to_none = True)
            torch.cuda.empty_cache()

            if iteration % args.vis_step == 0 or iteration == 1:
                other_img = []
                feature = render_pkg['feature'] / alpha.clamp_min(1e-5)
                t_map = feature[0:1]
                v_map = feature[1:]
                v_norm_map = v_map.norm(dim=0, keepdim=True)

                et_color = visualize_depth(t_map, near=0.01, far=1)
                v_color = visualize_depth(v_norm_map, near=0.01, far=1)
                other_img.append(et_color)
                other_img.append(v_color)

                if viewpoint_cam.pts_depth is not None:
                    pts_depth_vis = visualize_depth(viewpoint_cam.pts_depth)
                    other_img.append(pts_depth_vis)

                grid = make_grid([
                    image, 
                    gt_image, 
                    alpha.repeat(3, 1, 1),
                    torch.logical_not(sky_mask[:1]).float().repeat(3, 1, 1),
                    visualize_depth(depth), 
                ] + other_img, nrow=4)

                save_image(grid, os.path.join(vis_path, f"{iteration:05d}_{viewpoint_cam.colmap_id:03d}.png"))
            
            if iteration % args.scale_increase_interval == 0:
                scene.upScale()

            if iteration in args.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((env_map.capture(), iteration), scene.model_path + "/env_light_chkpnt" + str(iteration) + ".pth")


def complete_eval(tb_writer, iteration, test_iterations, scene : Scene, renderFunc, renderArgs, log_dict, env_map=None):
    from lpipsPyTorch import lpips

    if tb_writer:
        for key, value in log_dict.items():
            tb_writer.add_scalar(f'train/{key}', value, iteration)

    if iteration in test_iterations:
        scale = scene.resolution_scales[scene.scale_index]
        if iteration < args.iterations:
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},)
        else:
            if "kitti" in args.model_path:
                # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
                num = len(scene.getTrainCameras())//2
                eval_train_frame = num//5
                traincamera = sorted(scene.getTrainCameras(), key =lambda x: x.colmap_id)
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                    {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                                {'name': 'train', 'cameras': scene.getTrainCameras()})



        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
                os.makedirs(outdir,exist_ok=True)
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = render_pkg['depth']
                    alpha = render_pkg['alpha']
                    sky_depth = 900
                    depth = depth / alpha.clamp_min(EPS)
                    if env_map is not None:
                        if args.depth_blend_mode == 0:  # harmonic mean
                            depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                        elif args.depth_blend_mode == 1:
                            depth = alpha * depth + (1 - alpha) * sky_depth
                
                    depth = visualize_depth(depth)
                    alpha = alpha.repeat(3, 1, 1)

                    grid = [gt_image, image, alpha, depth]
                    grid = make_grid(grid, nrow=2)

                    save_image(grid, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

                    l1_test += F.l1_loss(image, gt_image).double()
                    psnr_test += psnr(image, gt_image).double()
                    ssim_test += ssim(image, gt_image).double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').double()  # very slow

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                with open(os.path.join(outdir, "metrics.json"), "w") as f:
                    json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item(), "ssim": ssim_test.item(), "lpips": lpips_test.item()}, f)
        torch.cuda.empty_cache()


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
    print(args)
    
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if args.exhaust_test:
        args.test_iterations += [i for i in range(0,args.iterations, args.test_interval)]
    
    print("Optimizing " + args.model_path)

    seed_everything(args.seed)

    training(args)

    # All done
    print("\nTraining complete.")
