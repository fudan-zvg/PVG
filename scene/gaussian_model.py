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
import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_step_lr_func
from torch import nn
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self._velocity = torch.empty(0)
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.t_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.time_duration = args.time_duration
        self.no_time_split = args.no_time_split
        self.t_grad = args.t_grad
        self.contract = args.contract
        self.t_init = args.t_init
        self.big_point_threshold = args.big_point_threshold

        self.T = args.cycle 
        self.velocity_decay = args.velocity_decay
        self.random_init_point = args.random_init_point

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t,
            self._scaling_t,
            self._velocity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.t_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.T,
            self.velocity_decay,
        )

    def restore(self, model_args, training_args=None):
        (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._t,
            self._scaling_t,
            self._velocity,
            self.max_radii2D,
            xyz_gradient_accum,
            t_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.T,
            self.velocity_decay,
        ) = model_args
        self.setup_functions()
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_t_activation(self._scaling_t)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz_SHM(self, t):
        a = 1/self.T * np.pi * 2
        return self._xyz + self._velocity*torch.sin((t-self._t)*a)/a

    @property
    def get_inst_velocity(self):
        return self._velocity*torch.exp(-self.get_scaling_t/self.T/2*self.velocity_decay)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2

    def get_marginal_t(self, timestamp):
        return torch.exp(-0.5 * (self.get_t - timestamp) ** 2 / self.get_scaling_t ** 2)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        ## random up and far
        r_max = 100000
        r_min = 2
        num_sph = self.random_init_point

        theta = 2*torch.pi*torch.rand(num_sph)
        phi = (torch.pi/2*0.99*torch.rand(num_sph))**1.5 # x**a decay
        s = torch.rand(num_sph)
        r_1 = s*1/r_min+(1-s)*1/r_max
        r = 1/r_1
        pts_sph = torch.stack([r*torch.cos(theta)*torch.cos(phi), r*torch.sin(theta)*torch.cos(phi), r*torch.sin(phi)],dim=-1).cuda()

        r_rec = r_min
        num_rec = self.random_init_point
        pts_rec = torch.stack([r_rec*(torch.rand(num_rec)-0.5),r_rec*(torch.rand(num_rec)-0.5),
                               r_rec*(torch.rand(num_rec))],dim=-1).cuda()

        pts_sph = torch.cat([pts_rec, pts_sph], dim=0)
        pts_sph[:,2] = -pts_sph[:,2]+1

        fused_point_cloud = torch.cat([fused_point_cloud, pts_sph], dim=0)
        features = torch.cat([features,
                              torch.zeros([pts_sph.size(0), features.size(1), features.size(2)]).float().cuda()],
                             dim=0)

        if pcd.time is None or pcd.time.shape[0] != fused_point_cloud.shape[0]:
            if pcd.time is None:
                time = (np.random.rand(pcd.points.shape[0], 1) * 1.2 - 0.1) * (
                        self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                time = pcd.time

            if self.t_init < 1:
                random_times = (torch.rand(fused_point_cloud.shape[0]-pcd.points.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                        self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
                pts_times = torch.from_numpy(time.copy()).float().cuda()
                fused_times = torch.cat([pts_times, random_times], dim=0)
            else:
                fused_times = torch.full_like(fused_point_cloud[..., :1],
                                                0.5 * (self.time_duration[1] + self.time_duration[0]))
        else:
            fused_times = torch.from_numpy(np.asarray(pcd.time.copy())).cuda().float()
            fused_times_sh = torch.full_like(pts_sph[..., :1], 0.5 * (self.time_duration[1] + self.time_duration[0]))
            fused_times = torch.cat([fused_times, fused_times_sh], dim=0)

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        dist_t = torch.full_like(fused_times, (self.time_duration[1] - self.time_duration[0])*self.t_init)
        scales_t = self.scaling_t_inverse_activation(torch.sqrt(dist_t))
        velocity = torch.full((fused_point_cloud.shape[0], 3), 0., device="cuda")
        
        opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._velocity = nn.Parameter(velocity.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._t], 'lr': training_args.t_lr_init, "name": "t"},
            {'params': [self._scaling_t], 'lr': training_args.scaling_t_lr, "name": "scaling_t"},
            {'params': [self._velocity], 'lr': training_args.velocity_lr * self.spatial_lr_scale, "name": "velocity"},
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

        final_decay = training_args.position_lr_final / training_args.position_lr_init

        self.t_scheduler_args = get_expon_lr_func(lr_init=training_args.t_lr_init,
                                                    lr_final=training_args.t_lr_init * final_decay,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "t":
                lr = self.t_scheduler_args(iteration)
                param_group['lr'] = lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        self._velocity = optimizable_tensors['velocity']
        self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_velocity):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "t": new_t,
             "scaling_t": new_scaling_t,
             "velocity": new_velocity,
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        self._velocity = optimizable_tensors['velocity']
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2, time_split=False,
                          joint_sample=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1 # -0
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent*scale_factor)
        decay_factor = N*0.8
        if not self.no_time_split:
            N = N+1

        if time_split:
            padded_grad_t = torch.zeros((n_init_points), device="cuda")
            padded_grad_t[:grads_t.shape[0]] = grads_t.squeeze()
            selected_time_mask = torch.where(padded_grad_t >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense

            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values > extend_thresh)
            if joint_sample:
                selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (decay_factor))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        xyz = self.get_xyz[selected_pts_mask]

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz.repeat(N, 1)

        new_t = None
        new_scaling_t = None
        new_velocity = None
        stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
        means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
        samples_t = torch.normal(mean=means_t, std=stds_t)
        new_t = samples_t+self.get_t[selected_pts_mask].repeat(N, 1)

        new_scaling_t = self.scaling_t_inverse_activation(
            self.get_scaling_t[selected_pts_mask].repeat(N, 1)/ (decay_factor))



        new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)

        new_xyz = new_xyz + self.get_inst_velocity[selected_pts_mask].repeat(N, 1) * (samples_t)

        not_split_xyz_mask =  torch.max(self.get_scaling[selected_pts_mask], dim=1).values < \
                                self.percent_dense * scene_extent*scale_factor[selected_pts_mask]
        new_scaling[not_split_xyz_mask.repeat(N)] = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1))[not_split_xyz_mask.repeat(N)]

        if time_split:
            not_split_t_mask = self.get_scaling_t[selected_pts_mask].squeeze() < extend_thresh
            new_scaling_t[not_split_t_mask.repeat(N)] = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1))[not_split_t_mask.repeat(N)]

        if self.no_time_split:
            new_scaling_t = self.scaling_t_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_t, new_scaling_t, new_velocity)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, time_clone=False):
        t_scale_factor=self.get_scaling_t.clamp(0,self.T)
        t_scale_factor=torch.exp(-t_scale_factor/self.T).squeeze()

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*scene_extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/scene_extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/scene_extent

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values <= self.percent_dense * scene_extent*scale_factor)
        if time_clone:
            selected_time_mask = torch.where(torch.norm(grads_t, dim=-1) >= grad_t_threshold, True, False)
            extend_thresh = self.percent_dense
            selected_time_mask = torch.logical_and(selected_time_mask,
                                                   torch.max(self.get_scaling_t, dim=1).values <= extend_thresh)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_time_mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_velocity = None
        new_t = self._t[selected_pts_mask]
        new_scaling_t = self._scaling_t[selected_pts_mask]
        new_velocity = self._velocity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_t, new_scaling_t, new_velocity)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            grads_t = self.t_gradient_accum / self.denom
            grads_t[grads_t.isnan()] = 0.0

            if self.t_grad:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=True)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=True)
            else:
                self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t, time_clone=False)
                self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t, time_split=False)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if self.contract:
            scale_factor = self._xyz.norm(dim=-1)*extent-1
            scale_factor = torch.where(scale_factor<=1, 1, scale_factor)/extent
        else:
            scale_factor = torch.ones_like(self._xyz)[:,0]/extent

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > self.big_point_threshold * extent * scale_factor  ## ori 0.1
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
        self.t_gradient_accum[update_filter] += self._t.grad.clone()[update_filter]
