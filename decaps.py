import torch
import torch.nn as nn
import torch.nn.functional as F
from config import options
from models import *
from models.custom import CustomNet
from models.senet import *
from utils.other_utils import squash, coord_addition
import numpy as np


class SimNet(nn.Module):
    def __init__(self, args):
        super(SimNet, self).__init__()
        self.args = args

        self.sim_net = nn.Sequential(
            nn.Linear(options.digit_cap_dim, args.h1),
            nn.ReLU(inplace=True),
            nn.Linear(args.h1, args.h2),
            nn.ReLU(inplace=True),
            nn.Linear(args.h2, args.img_h * args.img_w),
            nn.Sigmoid()
        )

    def forward(self, imgs):
        x = self.sim_net()
        return x


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias))

    def forward(self, x):
        return self.net(x)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, cap_dim, num_cap_map):
        super(PrimaryCapsLayer, self).__init__()

        self.capsule_dim = cap_dim
        self.num_cap_map = num_cap_map
        self.conv_out = Conv2dSame(in_channels, out_channels, kernel_size)

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.conv_out(x)
        map_dim = outputs.size(-1)
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, map_dim, map_dim)
        # [bs, 8 (or 10), 32, 6, 6]
        outputs = outputs.view(batch_size, self.capsule_dim, self.num_cap_map, -1).transpose(1, 2).transpose(2, 3)
        # [bs, 32, 36, 8]
        outputs = squash(outputs)
        return outputs


class DigitCapsLayer(nn.Module):
    def __init__(self, num_digit_cap, num_prim_cap, num_prim_map, in_cap_dim, out_cap_dim, num_iterations, use_simnet):
        super(DigitCapsLayer, self).__init__()
        self.num_prim_cap = num_prim_cap
        self.num_digit_cap = num_digit_cap
        self.num_iterations = num_iterations
        self.out_cap_dim = out_cap_dim
        self.use_simnet = use_simnet

        if options.share_weight:
            self.W = nn.Parameter(torch.randn(1, num_prim_map, 1, num_digit_cap, out_cap_dim, in_cap_dim))
            # [1, 32, 1, 10, 16, 8]
        else:
            self.W = nn.Parameter(torch.randn(1, num_prim_map, num_prim_cap, num_digit_cap, out_cap_dim, in_cap_dim))
            # [1, 32, 36, 10, 16, 8]

    def forward(self, x):
        batch_size, num_maps, num_in_caps, _ = x.size()          # [bs, 32, 36, 8]
        u = x[:, :, :, None, :, None]
        # [bs, 32, 36, 10, 8, 1]
        u_hat = torch.matmul(self.W, u)
        # [10, 32, 36, 10, 16, 1]

        if options.add_coord:
            u_hat = coord_addition(u_hat, norm_coord=options.norm_coord)   # [10, 32, 36, 10, 16, 1]

        b = torch.zeros(batch_size, num_maps, num_in_caps, self.num_digit_cap, 1, 1).cuda()
        # [bs, 32, 36, 10, 1, 1]
        for i in range(self.num_iterations):

            # c = F.softmax(b, dim=2)  # original is dim=3
            #
            c = F.softmax(b.view(batch_size, num_maps, num_in_caps*self.num_digit_cap, 1, 1), dim=2)  # original is dim=3
            c = c.view(batch_size, num_maps, num_in_caps, self.num_digit_cap, 1, 1)

            s = (c * u_hat).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
            # [10, 1, 1, 10, 16, 1]
            outputs = squash(s, dim=-2)

            if i != self.num_iterations - 1:
                outputs_tiled = outputs.repeat(1, num_maps, num_in_caps, 1, 1, 1)
                u_produce_v = torch.matmul(u_hat.transpose(-1, -2), outputs_tiled)
                b = b + u_produce_v

        map_size = int(np.sqrt(num_in_caps))

        x = (c * u_hat).view(batch_size, num_maps, map_size, map_size, self.num_digit_cap, self.out_cap_dim)
        # [bs, 32, 6, 6, 10, 16]
        c_maps = (x ** 2).sum(dim=-1) ** 0.5
        # c_maps = c.reshape(batch_size, num_maps, map_size, map_size, self.num_digit_cap)
        # (batch_size, 32, 6, 6, 10)

        features = (c * u_hat).mean(dim=2).squeeze(-1)  # [bs, 32, 10, 16]

        return outputs.squeeze(2).squeeze(1).squeeze(-1), c_maps, features


class CapsuleNet(nn.Module):
    def __init__(self, args):
        super(CapsuleNet, self).__init__()
        self.args = args

        # convolution layer
        if options.feature_extractor == 'resnet':
            net = resnet50(pretrained=True).get_features()
            self.num_features = 1024
            self.map_size = (20 * 20) / (args.s1 * args.s1)
        elif options.feature_extractor == 'densenet':
            net = densenet121(pretrained=True).get_features()
            self.num_features = 1024
            self.map_size = (6 * 6) / (args.s1 * args.s1)
        elif options.feature_extractor == 'inception':
            net = inception_v3(pretrained=True).get_features()
            net.aux_logits = False
            self.num_features = 768
            self.map_size = (30 * 30) / (args.s1 * args.s1)
        elif options.feature_extractor == 'custom':
            net = CustomNet()
            self.num_features = 1024
            # self.map_size = (18 * 18) / (args.s1 * args.s1)
        elif options.feature_extractor == 'senet':
            net = se_resnext50_32x4d().cuda()
            net = net.get_features()
            self.map_size = (24 * 24) / (args.s1 * args.s1)
            self.num_features = 1024

        self.features = net

        self.conv1 = Conv2dSame(self.num_features, args.f1, 1)

        # primary capsule layer
        assert args.f2 % args.primary_cap_dim == 0
        self.num_prim_map = int(args.f2 / args.primary_cap_dim)
        self.primary_capsules = PrimaryCapsLayer(in_channels=args.f1, out_channels=args.f2,
                                                 kernel_size=args.k2, stride=args.s1,
                                                 cap_dim=args.primary_cap_dim,
                                                 num_cap_map=self.num_prim_map)
        num_prim_cap = int((args.img_h - 2*(args.k2-1)) * (args.img_h - 2*(args.k2-1)) / (args.s1*args.s1))

        self.digit_capsules = DigitCapsLayer(num_digit_cap=args.num_classes,
                                             num_prim_cap=num_prim_cap,
                                             num_prim_map=self.num_prim_map,
                                             in_cap_dim=args.primary_cap_dim,
                                             out_cap_dim=args.digit_cap_dim,
                                             num_iterations=args.num_iterations,
                                             use_simnet=args.use_simnet)

        if args.add_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(16 * args.num_classes, args.h1),
                nn.ReLU(inplace=True),
                nn.Linear(args.h1, args.h2),
                nn.ReLU(inplace=True),
                nn.Linear(args.h2, args.img_h * args.img_w),
                nn.Sigmoid()
            )

    def forward(self, imgs, y=None):
        x = self.features(imgs)
        x = F.relu(self.conv1(x), inplace=True)
        x_child = self.primary_capsules(x)
        x, attention_maps, feats = self.digit_capsules(x_child)

        v_length = (x ** 2).sum(dim=-1) ** 0.5

        _, y_pred = v_length.max(dim=1)
        y_pred_ohe = F.one_hot(y_pred, self.args.num_classes)

        if y is None:
            y = y_pred_ohe

        img_reconst = torch.zeros_like(imgs)
        # if self.args.add_decoder:
        #     img_reconst = self.decoder((x * y[:, :, None].float()).view(x.size(0), -1))

        # Generate Attention Map
        batch_size, NUM_MAPS, H, W, _ = attention_maps.size()
        if self.training:
            # Randomly choose one of attention maps ck
            k_indices = np.random.randint(NUM_MAPS, size=batch_size)
            attention_map = attention_maps[torch.arange(batch_size), k_indices, :, :, y.argmax(dim=1)].to(torch.device("cuda"))
            # (B, H, W)
        else:
            attention_maps_cls = attention_maps[torch.arange(batch_size), :, :, :, y.argmax(dim=1)].to(torch.device("cuda"))
            # (B, NUM_MAPS, H, W)

            # Object Localization Am = mean(sum(Ak))
            attention_map = torch.mean(attention_maps_cls, dim=1, keepdim=True)  # (B, 1, H, W)

        # Normalize Attention Map
        attention_map = attention_map.view(batch_size, -1)  # (B, H * W)
        attention_map_max, _ = attention_map.max(dim=1, keepdim=True)  # (B, 1)
        attention_map_min, _ = attention_map.min(dim=1, keepdim=True)  # (B, 1)
        attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, H * W)
        attention_map = attention_map.view(batch_size, 1, H, W)  # (B, 1, H, W)

        return y_pred_ohe, img_reconst, v_length, attention_map, feats, attention_maps, x
