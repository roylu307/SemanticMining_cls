import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from pointnet2_ops import pointnet2_utils

import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
import os
from typing import List, Tuple
from scipy.stats import t as student_t
import statistics as stats
import math

# DensePoint: 2 PPools + 3 PConvs + 1 global pool; narrowness k = 24; group number g = 2
class DensePoint(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        # stage 1 begin
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.25],
                nsamples=[64],
                mlps=[[input_channels, 144]],
                use_xyz=use_xyz,
                pool=True
            )
        )
        # stage 1 end
        
        self.SA_modules.append(SphereConv(K=16, D=2, P=512, C=256, C_pts_fts=64, add_C=144, bottleneck_ratio=2, conv0=True, after_pool=False))
        self.SA_modules.append(SphereConv(K=16, D=2, P=512, C=256, C_pts_fts=64, add_C=272, bottleneck_ratio=2, conv0=True, before_pool=True))
        # self.SA_modules.append(SphereConv(K=16, D=2, P=512, C=256, C_pts_fts=64, add_C=224, before_pool=False))

        # stage 2 begin
        input_channels = 400
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[input_channels, 144]],
                use_xyz=use_xyz,
                pool=True
            )
        )

        ''' base model: K=16, D=2, P=128, C=256, C_pts_fts=64, add_C=96 '''
        self.SA_modules.append(SphereConv(K=16, D=2, P=128, C=256, C_pts_fts=64, add_C=144, bottleneck_ratio=2, conv0=True, after_pool=False))
        self.SA_modules.append(SphereConv(K=16, D=2, P=128, C=256, C_pts_fts=64, add_C=272, bottleneck_ratio=2, conv0=True,))
        self.SA_modules.append(SphereConv(K=16, D=2, P=128, C=256, C_pts_fts=64, add_C=400, bottleneck_ratio=2, conv0=True,))
        self.SA_modules.append(SphereConv(K=16, D=2, P=128, C=256, C_pts_fts=64, add_C=528, bottleneck_ratio=2, conv0=True, before_pool=True))
#         input_channels = 93
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=128,
#                 radii=[0.39],
#                 nsamples=[16],
#                 mlps=[[input_channels, 96]],
#                 group_number=2,
#                 use_xyz=use_xyz,
#                 after_pool=True
#             )
#         )
        
#         input_channels = 117
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=128,
#                 radii=[0.39],
#                 nsamples=[16],
#                 mlps=[[input_channels, 96]],
#                 group_number=2,
#                 use_xyz=use_xyz
#             )
#         )
        
#         input_channels = 141
#         self.SA_modules.append(
#             PointnetSAModuleMSG(
#                 npoint=128,
#                 radii=[0.39],
#                 nsamples=[16],
#                 mlps=[[input_channels, 96]],
#                 group_number=2,
#                 use_xyz=use_xyz,
#                 before_pool=True
#             )
#         )
        # stage 2 end
       
        # global pooling
        input_channels = 656
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[input_channels, 512], use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            FC(512, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            FC(512, 256, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            FC(256, num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        idx=0
        for module in self.SA_modules:
            # print('layer {} input dim xyz:{}, features {}'.format(idx,xyz.shape,features.shape if features is not None else None))
            xyz, features = module(xyz, features)
            # print('layer {} output dim xyz:{}, features {}'.format(idx,xyz.shape if xyz is not None else None,features.shape))
            idx+=1
            
        return self.FC_layer(features.squeeze(-1))


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool = False

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """

        all_features = 0
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        
        if self.npoint is not None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint) \
                      if self.pool else torch.from_numpy(np.arange(xyz.size(1))).int().cuda().repeat(xyz.size(0), 1)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = None
        
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            if not self.pool and self.npoint is not None:
                new_features = [new_features, features]
            new_features = self.mlps[i](new_features)   # (B, mlp[-1], npoint)
            all_features += new_features
        
        return new_xyz, all_features


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            mlps: List[List[int]],
            group_number = 1,
            use_xyz: bool = True,
            pool: bool = False,
            before_pool: bool = False,
            after_pool: bool = False,
            bias = True,
            init = nn.init.kaiming_normal_
    ):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)
        self.pool = pool
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        if pool:
            C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
            C_out = mlps[0][1]
            pconv = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = (1, 1), 
                                       stride = (1, 1), bias = bias)
            init(pconv.weight)
            if bias:
                nn.init.constant_(pconv.bias, 0)
            convs = [pconv]
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if npoint is None:
                self.mlps.append(GloAvgConv(C_in = mlp_spec[0], C_out = mlp_spec[1]))
            elif pool:
                self.mlps.append(PointConv(C_in = mlp_spec[0], C_out = mlp_spec[1], convs = convs))
            else:
                self.mlps.append(EnhancedPointConv(C_in = mlp_spec[0], C_out = mlp_spec[1], group_number = group_number, before_pool = before_pool, after_pool = after_pool))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            use_xyz=use_xyz
        )

class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

########## PointConv begin ############
class PointConv(nn.Module):
    '''
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    '''
    def __init__(self, C_in, C_out, convs=None):
        super(PointConv, self).__init__()
        self.bn = nn.BatchNorm2d(C_out)
        self.activation = nn.ReLU(inplace=True)
        self.pconv = convs[0]
        
    def forward(self, x): # x: (B, C_in, npoint, nsample)
        nsample = x.size(3)
        x = self.activation(self.bn(self.pconv(x)))
        return F.max_pool2d(x, kernel_size = (1, nsample)).squeeze(3)
########## PointConv   end ############

class SphereConv(nn.Module):
    def __init__(self, K, D, P, C, C_pts_fts, add_C, bottleneck_ratio=4, conv0=False, before_pool=False, after_pool=False):
        
        super(SphereConv, self).__init__()
        self.before_pool, self.after_pool = before_pool, after_pool
        self.K = K
        self.conv0 = conv0
        
        in_channels_fts_conv = C_pts_fts + add_C

        if conv0:
            self.conv0 = nn.Conv2d(in_channels=3, out_channels=C_pts_fts, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(C_pts_fts, momentum=0.98)
        else:
            self.fc1 = nn.Linear(3, C_pts_fts//2)
            self.bn1 = nn.BatchNorm2d(C_pts_fts//2, momentum=0.98)
            self.fc2 = nn.Linear(C_pts_fts//2, C_pts_fts)
            self.bn2 = nn.BatchNorm2d(C_pts_fts, momentum=0.98)

        self.maxpool = nn.MaxPool2d((1, int(K/D)), stride=(1, int(K/D)))
        self.conv1 = nn.Conv2d(in_channels=in_channels_fts_conv, out_channels=C, kernel_size=(1, D), groups=D, bias=True)
        self.conv2 = nn.Conv1d(in_channels = C, out_channels = math.floor(C/bottleneck_ratio), kernel_size = 1, bias = True)
        self.bn3 = nn.BatchNorm1d(C, momentum=0.98)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.shortcut = nn.Sequential(
            nn.Conv1d(add_C, math.floor(C/bottleneck_ratio), kernel_size=1, bias=False),
            nn.BatchNorm1d(math.floor(C/bottleneck_ratio)))

        if after_pool:
            self.bn4 = nn.BatchNorm2d(add_C, momentum=0.98)
        if before_pool:
            self.bn4 = nn.BatchNorm1d(int(C/bottleneck_ratio) + add_C, momentum=0.98)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        
    def forward(self, pts, fts_prev):
        N = pts.shape[0]
        point_num=pts.shape[1]
             
        indices = self.knn_indices_general(pts, pts, self.K, sort=True)
        pts_flipped = pts.transpose(1,2).contiguous()
        pts_grouped = pointnet2_utils.grouping_operation(pts_flipped, indices.int()) # [N, 3, P, K]
        fts_grouped = pointnet2_utils.grouping_operation(fts_prev, indices.int()) # [N, add_C, P, K]
        if self.after_pool:
            fts_grouped = self.bn4(fts_grouped)

        fts = pts_grouped - pts_flipped.unsqueeze(dim=-1) # [N ,3, P, K]
        
        if self.conv0:
            fts = self.relu(self.bn0(self.conv0(fts)))
        else:
            fts = fts.transpose(1,2).transpose(2,3)
            fts = self.bn1(self.fc1(fts).transpose(1,3).transpose(2,3))
            fts = self.relu(fts).transpose(1,2).transpose(2,3)
            fts = self.bn2(self.fc2(fts).transpose(1,3).transpose(2,3))
            fts = self.relu(fts) # [N, C_pts_fts, P, K]
#         print(fts.shape)
        
        fts = torch.cat([fts, fts_grouped],dim=1) # [N, C_pts_fts+add_C, P, K]

        fts = self.maxpool(fts) # [N, C_pts_fts+add_C, P, D]
#         print(fts.shape)

        fts = self.conv1(fts) # [N, C_pts_fts+add_C, P, 1]
        fts = torch.squeeze(fts, dim=-1)
        fts = self.relu(self.bn3(fts))
        
        fts = self.dropout(self.conv2(fts)) # [N, C/4, P]
#         print(fts.shape)

        identity = self.shortcut(fts_prev)
        fts += identity


        fts = torch.cat((fts_prev, fts), dim=1)
        if self.before_pool:
            fts = self.relu(self.bn4(fts))
        
        return pts, fts
    
    def batch_distance_matrix_general(self, A, B):
        r_A = torch.sum(A * A, dim=2, keepdim=True)
        r_B = torch.sum(B * B, dim=2, keepdim=True)
        m = torch.matmul(A, torch.transpose(B, 1,2))
        D = r_A - 2 * m + torch.transpose(r_B, 2, 1)
        return D

    def knn_indices_general(self, queries, points, k, sort=True, unique=False):
        batch_size = queries.shape[0]
        point_num = queries.shape[1]
        D = self.batch_distance_matrix_general(queries, points)
        _, indices = torch.topk(-D, k=k, sorted=True)  # (N, P, K)
        # batch_indices = torch.arange(batch_size, device='cuda',dtype=int).view(-1, 1, 1, 1).repeat(1, point_num, k, 1)#.to('cuda')
        # indices = torch.cat((batch_indices, torch.unsqueeze(point_indices[:,:,tmp_k:],dim=3)), dim=3)
        return indices


########## EnhancedPointConv begin ############
class EnhancedPointConv(nn.Module):
    '''
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    '''
    def __init__(self, C_in, C_out, group_number=1, before_pool=False, after_pool=False, init=nn.init.kaiming_normal_, bias=True):
        super(EnhancedPointConv, self).__init__()
        self.before_pool, self.after_pool = before_pool, after_pool
        C_small = math.floor(C_out/4)
        self.conv_phi = nn.Conv2d(in_channels = C_in, out_channels = C_out, groups = group_number, kernel_size = (1, 1),
                                  stride = (1, 1), bias = bias)    # ~\phi function: grouped version
        self.conv_psi = nn.Conv1d(in_channels = C_out, out_channels = C_small, kernel_size = 1,
                              stride = 1, bias = bias)             # \psi function
        if not after_pool:
            self.bn_cin = nn.BatchNorm2d(C_in)
        self.bn_phi = nn.BatchNorm2d(C_out)
        if before_pool:
            self.bn_concat = nn.BatchNorm1d(C_in-3+C_small)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        init(self.conv_phi.weight)
        init(self.conv_psi.weight)
        if bias:
            nn.init.constant_(self.conv_phi.bias, 0)
            nn.init.constant_(self.conv_psi.bias, 0)

    def forward(self, input): # x: (B, C_in, npoint, nsample)
        x, last_feat = input[0], input[1]
        nsample = x.size(3)
        if not self.after_pool:
            x = self.activation(self.bn_cin(x))
        x = self.activation(self.bn_phi(self.conv_phi(x)))
        x = F.max_pool2d(x, kernel_size=(1, nsample)).squeeze(3)
        x = torch.cat((last_feat, self.dropout(self.conv_psi(x))), dim=1)
        
        if self.before_pool:
            x = self.activation(self.bn_concat(x))
        return x

########## EnhancedPointConv end ############


########## global convolutional pooling begin ############
class GloAvgConv(nn.Module):
    '''
    Input shape: (B, C_in, 1, nsample)
    Output shape: (B, C_out, npoint)
    '''
    def __init__(
            self, 
            C_in, 
            C_out, 
            init=nn.init.kaiming_normal_, 
            bias = True,
            activation = nn.ReLU(inplace=True)
    ):
        super(GloAvgConv, self).__init__()

        self.conv_avg = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias) 
        self.bn_avg = nn.BatchNorm2d(C_out)
        self.activation = activation
        
        init(self.conv_avg.weight)
        if bias:
            nn.init.constant_(self.conv_avg.bias, 0)
        
    def forward(self, x):
        nsample = x.size(3)
        x = self.activation(self.bn_avg(self.conv_avg(x)))
        x = F.max_pool2d(x, kernel_size = (1, nsample)).squeeze(3)
        return x
########## global convolutional pooling end ############

class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)