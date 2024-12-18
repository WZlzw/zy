import math
import torch
import torch.nn as nn
from functools import partial


# 输入输出相同
class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size//2,
            ),
            norm(out_features),
            act(),
        )

Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x):
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


class MBConv(nn.Sequential):
    expansion = 1
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features*expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, 
                                      expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, 
                                      expanded_features, 
                                      groups=expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
        

class SEMBConv(nn.Sequential):
    expansion = 1
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features*expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        # narrow -> wide
                        Conv1X1BnReLU(in_features, 
                                      expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # wide -> wide
                        Conv3X3BnReLU(expanded_features, 
                                      expanded_features, 
                                      groups=expanded_features,
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        SElayer(expanded_features),
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup  # oup：论文Fig2(a)中output的通道数
        init_channels = math.ceil(oup / ratio)  # init_channels: 在论文Fig2(b)中,黄色部分的通道数
                                                # ceil函数：向上取整，
                                                # ratio：在论文Fig2(b)中，output通道数与黄色部分通道数的比值
        new_channels = init_channels*(ratio-1)  # new_channels: 在论文Fig2(b)中，output红色部分的通道数

        self.primary_conv = nn.Sequential(      # 输入所用的普通的卷积运算，生成黄色部分
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                                                # 1//2=0
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(   # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                                                # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)         # torch.cat: 在给定维度上对输入的张量序列进行连接操作
                                                # 将黄色部分和红色部分在通道上进行拼接
        return out[:,:self.oup,:,:]             # 输出Fig2中的output；由于向上取整，可以会导致通道数大于self.out


if __name__ == '__main__':
    x = torch.randn(1, 32, 24, 24)
    print(SEMBConv(32, 32)(x).shape)