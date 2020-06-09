import lib.network as dlib
import lib.foldingnet as flib
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedEncode(dlib.Encode):
    def __init__(self, *args, **kwargs):
        super(ModifiedEncode, self).__init__(*args, **kwargs)


class ModifiedRecon(flib.Recon):
    def __init__(self, num_points, *args, **kwargs):
        assert num_points <= 1944
        super(ModifiedRecon, self).__init__(*args, **kwargs)

        stride = 1944 // num_points
        self.grid = [self.grid[i] for i in range(0, 1944, stride)]
        self.grid = torch.stack(self.grid, dim=0)[:num_points]

        self.N = num_points

        self.register_buffer("grid_buf", self.grid)

        self.var = nn.Linear(num_points, 1)

    def forward(self, codeword):
        if self.training:
            # ADD VAE MODULE HERE
            noise = self.var(codeword)
            
            eps = torch.randn_like(noise)
            codeword = (codeword + torch.exp(noise / 2.0) * eps)
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(noise) + codeword ** 2 - 1.0 - noise, 1))
            return super().forward(codeword), kl_loss
        else:
            return super().forward(codeword)

class ModifiedPose(dlib.Pose):
    def __init__(self, *args, **kwargs):
        super(ModifiedPose, self).__init__(*args, **kwargs)

        self.conv1_r = torch.nn.Conv1d(1408 * 2, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408 * 2, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408 * 2, 640, 1)


class ModifiedFoldingNetShapes(nn.Module):
    def __init__(self, num_points, MLP_dims, FC_dims, Folding1_dims, Folding2_dims, MLP_doLastRelu):
        super(ModifiedFoldingNetShapes, self).__init__()

        self.encoding = ModifiedEncode(num_points)

        self.reconstructing = ModifiedRecon(
            num_points, Folding1_dims, Folding2_dims)

        # self.var = nn.Linear(num_points, 1)

    def encode(self, img, x, choose):
        return self.encoding(img, x, choose)

    def recon(self, codeword):
        return self.reconstructing(codeword)
        # if self.training:
        #     # ADD VAE MODULE HERE
        #     noise = self.var(codeword)
            
        #     eps = torch.randn_like(noise)
        #     codeword = (codeword + torch.exp(noise / 2.0) * eps)
        #     kl_loss = torch.mean(0.5 * torch.sum(torch.exp(noise) + codeword ** 2 - 1.0 - noise, 1))
        #     return self.reconstructing(codeword), kl_loss
        # else:
        #     return self.reconstructing(codeword)


class ModifiedPoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(ModifiedPoseNet, self).__init__()

        self.encoding = ModifiedEncode(num_points)

        self.posing = ModifiedPose(num_points, num_obj)

    def encode(self, img, x, choose):
        return self.encoding(img, x, choose)

    def pose(self, codeword, obj):
        return self.posing(codeword, obj)


class ModifiedPoseRefineNet(dlib.PoseRefineNet):
    def __init__(self, *args, **kwargs):
        super(ModifiedPoseRefineNet, self).__init__(*args, **kwargs)


class CASS(nn.Module):
    def __init__(self, opt):
        super().__init__()

        MLP_dims = (3, 64, 64, 64, 128, 1024)
        FC_dims = (1024, 512, 1408)
        Folding1_dims = (1408+9, 512, 512, 3)
        Folding2_dims = (1408+3, 512, 512, 3)
        MLP_doLastRelu = False
        self.opt = opt
        self.estimator = ModifiedPoseNet(
            num_points=opt.num_points, num_obj=opt.num_objects
        )
        self.refiner = ModifiedPoseRefineNet(
            num_points=opt.num_points, num_obj=opt.num_objects
        )
        self.foldingnet = ModifiedFoldingNetShapes(
            opt.num_points,
            MLP_dims, FC_dims, Folding1_dims, Folding2_dims, MLP_doLastRelu
        )
