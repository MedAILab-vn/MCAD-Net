import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

'''
For testing only
'''


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class EMA(nn.Module):
    def __init__(self, in_planes, factor=8):
        super(EMA, self).__init__()
        channels = in_planes
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class LSKNet_Attention(nn.Module):
    def __init__(self, in_planes):
        super(LSKNet_Attention, self).__init__()
        self.in_planes = in_planes
        self.dwconv1 = nn.Conv2d(in_planes, in_planes, kernel_size=5, padding=2, groups=in_planes)
        self.dwconv2 = nn.Conv2d(in_planes, in_planes, kernel_size=7, stride=1, padding=9, dilation=3, groups=in_planes)
        self.conv1x1_1 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_planes, in_planes, kernel_size=1)
        self.conv_spatial = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv_fuse = nn.Conv2d(in_planes, in_planes, kernel_size=1)

    def forward(self, x):
        u1 = self.dwconv1(x)
        u2 = self.dwconv2(u1)
        u1_tilde = self.conv1x1_1(u1)
        u2_tilde = self.conv1x1_2(u2)
        u_cat = torch.cat([u1_tilde, u2_tilde], dim=1)
        sa_avg = torch.mean(u_cat, dim=1, keepdim=True)
        sa_max = torch.max(u_cat, dim=1, keepdim=True)[0]
        sa_cat = torch.cat([sa_avg, sa_max], dim=1)
        sa_maps = self.sigmoid(self.conv_spatial(sa_cat))
        sa1, sa2 = torch.chunk(sa_maps, 2, dim=1)
        s_sum = sa1 * u1_tilde + sa2 * u2_tilde
        out = x * self.conv_fuse(s_sum)
        return out

class FS_Block(nn.Module):
    def __init__(self, in_planes):
        super(FS_Block, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.fusion(x)

class ECANet(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1):
        super(ECANet, self).__init__()
        k = int(abs((math.log(in_planes, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class GAM_Attention(nn.Module):
    def __init__(self, in_planes, rate=4):
        super(GAM_Attention, self).__init__()
        in_channels = in_planes
        out_channels = in_planes

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_channel_att = self.channel_attention(x_permute).view(b, h, w, c)

        x_channel_att = x_channel_att.permute(0, 3, 1, 2)
        x_channel_att = torch.sigmoid(x_channel_att)
        out_c = x * x_channel_att

        x_spatial_att = self.spatial_attention(out_c)
        x_spatial_att = torch.sigmoid(x_spatial_att)
        out = out_c * x_spatial_att

        return out

class TripletAM(nn.Module):
    def __init__(self, in_planes):
        super(TripletAM, self).__init__()
    def forward(self, x): return x

class UFFN(nn.Module):
    def __init__(self, in_planes, out_dim=None, expansion_rate=4., act_layer=nn.GELU, use_mid_conv=True):
        super(UFFN, self).__init__()
        dim = in_planes
        if out_dim is None:
            out_dim = in_planes

        if act_layer is None:
            act_layer = nn.GELU

        hidden_dim = int(dim * expansion_rate)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        )
        self.mid_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            act_layer()
        ) if use_mid_conv else nn.Identity()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        net = self.fc1(x)
        net = self.mid_conv(net)
        net = self.fc2(net)
        return net

class Stable(nn.Module):
    def __init__(self, in_planes):
        super(Stable, self).__init__()
        self.cross_fusion_x = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        )

    def forward(self, x):
        return self.cross_fusion_x(x)

def build_attention_block(attention_type, in_planes):
    if attention_type == 'CBAM':
        return CBAM(in_planes=in_planes)
    elif attention_type == 'ECA':
        return ECANet(in_planes=in_planes)
    elif attention_type == 'GAM':
        return GAM_Attention(in_planes=in_planes)
    elif attention_type == 'TripletAM':
        return TripletAM(in_planes=in_planes)
    elif attention_type == 'EMA':
        return EMA(in_planes=in_planes)
    elif attention_type == 'LSKNet':
        return LSKNet_Attention(in_planes=in_planes)
    elif attention_type == 'FS':
        return FS_Block(in_planes=in_planes)
    elif attention_type == 'UFFN':
        return UFFN(in_planes=in_planes, use_mid_conv=False)
    elif attention_type == 'Stable':
        return Stable(in_planes=in_planes)
    elif attention_type == 'None' or attention_type == None :
        return nn.Identity()
    else:
        raise ValueError(f"[ERROR] Attention architecture not found: '{attention_type}'. Please check config.py!")