# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2

from update import BasicUpdateBlock
from corr import CorrBlock
from utils.utils import coords_grid, InputPadder
from extractor import ResNetFPN
from layer import conv1x1, conv3x3

from huggingface_hub import PyTorchModelHubMixin

class FlowSeek(
    nn.Module,
    PyTorchModelHubMixin
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2

        self.da_size = args.da_size

        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)

        self.da_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        }

        self.dav2 = DepthAnythingV2(**self.da_model_configs[args.da_size])
        import os as _os
        _dav2_path = _os.environ.get(
            'FLOWSEEK_DAV2_CHECKPOINT',
            f'weights/depth_anything_v2_{args.da_size}.pth'
        )
        self.dav2.load_state_dict(torch.load(_dav2_path, map_location='cpu', weights_only=False))
        self.dav2 = self.dav2.cuda().eval()
        for param in self.dav2.parameters():
                param.requires_grad = False              

        self.merge_head = nn.Sequential(
            nn.Conv2d(self.da_model_configs[args.da_size]['features'], self.da_model_configs[args.da_size]['features']//2*3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.da_model_configs[args.da_size]['features']//2*3, self.da_model_configs[args.da_size]['features']*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.da_model_configs[args.da_size]['features']*2, self.da_model_configs[args.da_size]['features']*2, 3, stride=2, padding=1),
        )

        self.bnet = ResNetFPN(args, input_dim=16, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim*2, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim*2, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim*2, cdim=args.dim*2)


    def create_bases(self, disp):
        B, C, H, W = disp.shape
        assert C == 1
        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1).cuda()
        v = v.repeat(B, 1, 1, 1).cuda()

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)
        
        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
        Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)
        
        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1) # Bx(8x2)xHxW
        return M
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords2 = coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)


    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False, demo=False):
        """ Estimate optical flow between pair of frames """
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        image1_res = F.interpolate(image1, (518, 518), mode="bilinear", align_corners = False) / 255. 
        image2_res = F.interpolate(image2, (518, 518), mode="bilinear", align_corners = False) / 255.

        mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()
        std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).unsqueeze(0).unsqueeze(2).unsqueeze(2).cuda()
        image1_res = image1_res / mean - std # should be (image1_res - mean) / std. Models were trained with image1_res / mean - std, switching to the correct normalization alters EPE on the second digit 
        image2_res = image2_res / mean - std # should be (image2_res - mean) / std. Models were trained with image2_res / mean - std, switching to the correct normalization alters EPE on the second digit

        im1_path1, depth1 = self.dav2.forward(image1_res.float())
        im2_path1, _ = self.dav2.forward(image2_res.float())

        im1_path1 = F.interpolate(im1_path1, (H, W), mode="bilinear", align_corners = False)
        im2_path1 = F.interpolate(im2_path1, (H, W), mode="bilinear", align_corners = False)
        bases1 = self.create_bases(F.interpolate(depth1, (H, W), mode="bilinear", align_corners = False))            
        
        mono1 = self.merge_head(im1_path1)
        mono2 = self.merge_head(im2_path1)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        bases1 = padder.pad(bases1)
        
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H//8, W//8, device=image1.device)
        
        # run the context network
        cnet_inputs = torch.cat([image1, image2], dim=1)
        # if self.use_da:
            

        cnet = self.cnet(cnet_inputs)
        
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)
                
        bnet_inputs = bases1[0]
        bnet = self.bnet(bnet_inputs)
        bnet = self.init_conv(bnet)
        netbases, ctxbases = torch.split(bnet, [self.args.dim, self.args.dim], dim=1)

        context = torch.cat((context, ctxbases), 1)
        net = torch.cat((net, netbases), 1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
            
        if self.args.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)

            fmap1_8x = torch.cat((fmap1_8x,mono1), 1)
            fmap2_8x = torch.cat((fmap2_8x,mono2), 1)

            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        if test_mode == False:
            # exlude invalid pixels and extremely large diplacements
            nf_predictions = []
            for i in range(len(info_predictions)):
                if not self.args.use_var:
                    var_max = var_min = 0
                else:
                    var_max = self.args.var_max
                    var_min = self.args.var_min
                    
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                # Large b Component                
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
                # Small b Component
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
                # term2: [N, 2, m, H, W]
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                # term1: [N, m, H, W]
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)

            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}
