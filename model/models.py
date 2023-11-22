from torch import nn
import torch
from pytorch_wavelets import DWTForward
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from efficientnet_pytorch import EfficientNet

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


# this part is for wavelet data generation


class WaveGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.DWT1 = DWTForward(J=1, mode='symmetric', wave='db2')
        self.DWT2 = DWTForward(J=1, mode='symmetric', wave='coif1')
        self.DWT3 = DWTForward(J=1, mode='symmetric', wave='sym2')
        self.DWTs = [self.DWT1, self.DWT2, self.DWT3]

    '''
    :method description: gain the DWT descompose of X, only keep the high frequency coefficients
    ----------------------
    :param: x
    ----------------------
    :return 1 layer wavelet decompose of x 
    '''

    def DWT(self, x: torch.Tensor):
        size = int(x.shape[-1]/2)
        batch = x.shape[0]
        channel = x.shape[1]
        # 当channel==3时，即channel与小波种类不相关
        if channel == 3:

            x_H = [f(x)[1][0][..., :size, :size] for f in self.DWTs]

            x_L = [f(x)[0][..., :size, :size] for f in self.DWTs]

        # 当channel!=3,按channel拆分后再进行小波
        else:
            channel_pw = int(channel/3)
            splits = torch.split(x, channel_pw, 1)
            x_H = [f(splits[index])[1][0][..., :size, :size]
                   for index, f in enumerate(self.DWTs)]
            x_L = [f(splits[index])[0][..., :size, :size]
                   for index, f in enumerate(self.DWTs)]

        x_H = [torch.reshape(item, (batch, -1, size, size)) for item in x_H]
        x_L = [torch.reshape(item, (batch, -1, size, size)) for item in x_L]
        x_H = torch.cat(tuple(x_H), dim=1)
        x_L = torch.cat(tuple(x_L), dim=1)

        return x_L, x_H

    def features(self, x: torch.Tensor):
        # get the Tensor first, Tensor size=[B,C,D(HVD),H,W]
        x_low, x_high = self.DWT(x)
        # x = self.DWT(x)

        return x_low, x_high

    def forward(self, x):
        x_low, x_high = self.features(x)
        return x_low, x_high


# this branch only work with wavelet
class ProcessNet_wave(nn.Module):
    def __init__(self, waves_n):
        super().__init__()

        # the argument I need
        in_channel = waves_n * 9
        self.in_channel = in_channel
        # the module I need

        # Head
        self.DWT = WaveGen()
        # self.entry_l1 = nn.Conv2d(
        #     in_channel, in_channel*3, 3, 1, 1, bias=False)

        # self.relu = nn.ReLU(inplace=True)

        # middle flow
        # Block缩小尺寸的方法是池化
        self.bn1 = nn.BatchNorm2d(27)
        self.block1 = Block(
            27, 54, 4, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            54, 27, 2, 1, start_with_relu=False, grow_first=True)

        # 此时需要concat l2的DWT
        self.bn2 = nn.BatchNorm2d(108)
        self.block3 = Block(
            135, 270, 4, 2, start_with_relu=False, grow_first=True)
        self.block4 = Block(
            270, 135, 2, 1, start_with_relu=False, grow_first=True)

        self.bn3 = nn.BatchNorm2d(432)
        self.block5 = Block(
            567, 1134, 4, 2, start_with_relu=False, grow_first=True)
        self.block6 = Block(
            1134, 567, 2, 1, start_with_relu=False, grow_first=True)

        self.bn4 = nn.BatchNorm2d(567)
        self.block7 = Block(
            567, 1134, 4, 2, start_with_relu=False, grow_first=True)

        self.block8 = Block(
            1134, 1134, 2, 2, start_with_relu=False, grow_first=True)

        # self.block7 = Block(
        #     972, 486, 2, 1, start_with_relu=False, grow_first=True)
        # self.block8 = Block(
        #     486, 243, 2, 1, start_with_relu=False, grow_first=True)
        # self.block9 = Block(
        #     243, 128, 2, 1, start_with_relu=False, grow_first=True)

        # self.block10 = Block(
        #     128, 64, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     64, 32, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     32, 16, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     16, 8, 2, 1, start_with_relu=False, grow_first=True)
        self.compress = nn.Sequential(
            Block(
                972, 486, 2, 1, start_with_relu=False, grow_first=True),
            # Block(
            #     486, 243, 2, 1, start_with_relu=False, grow_first=True),



        )

        self.fea_compress = nn.Linear(1134, 1000)
        self.bn5 = nn.BatchNorm2d(1000)

    def features(self, input):
        x = input
        # 第一步 求一阶高频
        x_L, x_H = self.DWT(x)
        x = x_H
        x = self.bn1(x)

        # 第二步 对一阶段的高频系数进行卷积操作，
        x = self.block1(x)
        x = self.block2(x)

        # 第三步 求x的二阶段的高频系数
        x_LL, x_LH = self.DWT(x_L)
        x_HL, x_HH = self.DWT(x_H)
        x_t = torch.cat((x_LH, x_HH), dim=1)
        x_t = self.bn2(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # 第四部 对二阶高频做卷积
        x = self.block3(x)
        x = self.block4(x)

        # 第五步 求x的第三阶高频系数
        x_LLL, x_LLH = self.DWT(x_LL)
        x_LHL, x_LHH = self.DWT(x_LH)
        x_HLL, x_HLH = self.DWT(x_HL)
        x_HHL, x_HHH = self.DWT(x_HH)
        x_t = torch.cat((x_LLH, x_LHH, x_HLH, x_HHH), dim=1)
        x_t = self.bn3(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # 第六步 对三阶高频做卷积

        x = self.block5(x)
        x = self.block6(x)

        x = self.bn4(x)
        x = self.block7(x)
        x = self.block8(x)

        # 第七步 开始通道压缩,
        # x = self.compress(x)
        # x = self.block6(x)
        # x = self.block7(x)

        # x = self.block8(x)
        # x = self.block9(x)

        # 第八步，自适应地求一个1*1最大池化
        # ?改进空间
        x = F.adaptive_max_pool2d(x, (1, 1))

        # 第九步 x展开成长条
        x = x.view(x.size(0), -1)

        # 第十步，最终该分支就会输出对应的特征
        x = self.fea_compress(x)
        x = self.bn5(x)
        return x

    def forward(self, input):
        # step 1 turn input to level1 DWT high frequency coefficients
        x = input
        fea = self.features(x)

        return fea

# this branch only work with wavelet


class ProcessNet_wave_224(nn.Module):
    def __init__(self, waves_n):
        super().__init__()

        # the argument I need
        in_channel = waves_n * 9
        self.in_channel = in_channel
        # the module I need

        # Head
        self.DWT = WaveGen()
        # self.entry_l1 = nn.Conv2d(
        #     in_channel, in_channel*3, 3, 1, 1, bias=False)

        # self.relu = nn.ReLU(inplace=True)

        # middle flow
        # Block缩小尺寸的方法是池化
        self.bn1 = nn.BatchNorm2d(27)
        self.block1 = Block(
            27, 54, 2, 2, start_with_relu=False, grow_first=True)

        self.bn2 = nn.BatchNorm2d(108)

        self.block2 = Block(
            162, 324, 2, 2, start_with_relu=False, grow_first=True)

        # 此时需要concat l2的DWT
        self.bn3 = nn.BatchNorm2d(432)

        self.block3 = Block(
            756, 1512, 2, 2, start_with_relu=False, grow_first=True)

        self.block4 = Block(
            1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        self.block5 = Block(
            1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        self.block6 = Block(
            1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        self.block7 = Block(
            1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        self.block8 = Block(
            1512, 1512, 3, 1, start_with_relu=False, grow_first=True)

        self.block9 = Block(
            1512, 3024, 2, 2, start_with_relu=False, grow_first=True)

        self.bn4 = nn.BatchNorm2d(2000)

        # self.block7 = Block(
        #     972, 486, 2, 1, start_with_relu=False, grow_first=True)
        # self.block8 = Block(
        #     486, 243, 2, 1, start_with_relu=False, grow_first=True)
        # self.block9 = Block(
        #     243, 128, 2, 1, start_with_relu=False, grow_first=True)

        # self.block10 = Block(
        #     128, 64, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     64, 32, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     32, 16, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     16, 8, 2, 1, start_with_relu=False, grow_first=True)
        # self.compress = nn.Sequential(
        #     Block(
        #         972, 486, 2, 1, start_with_relu=False, grow_first=True),
        #     # Block(
        #     #     486, 243, 2, 1, start_with_relu=False, grow_first=True),

        # )

        self.fea_compress = nn.Linear(3024, 1000)
        self.bn5 = nn.BatchNorm2d(1000)

    def features(self, input):
        x = input
        # 第一步 求一阶高频
        x_L, x_H = self.DWT(x)
        x = x_H
        x = self.bn1(x)

        # 第二步 对一阶段的高频系数进行卷积操作，
        x = self.block1(x)
        # x = self.block2(x)

        # 第三步 求x的二阶段的高频系数
        x_LL, x_LH = self.DWT(x_L)
        x_HL, x_HH = self.DWT(x_H)
        x_t = torch.cat((x_LH, x_HH), dim=1)
        x_t = self.bn2(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)
        x = self.block2(x)

        # 第四部 对二阶高频做卷积
        # x = self.block3(x)
        # x = self.block4(x)

        # 第五步 求x的第三阶高频系数
        x_LLL, x_LLH = self.DWT(x_LL)
        x_LHL, x_LHH = self.DWT(x_LH)
        x_HLL, x_HLH = self.DWT(x_HL)
        x_HHL, x_HHH = self.DWT(x_HH)
        x_t = torch.cat((x_LLH, x_LHH, x_HLH, x_HHH), dim=1)
        x_t = self.bn3(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # 第六步 对三阶高频做卷积

        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.block9(x)

        # x = self.bn4(x)

        # 第七步 开始通道压缩,
        # x = self.compress(x)
        # x = self.block6(x)
        # x = self.block7(x)

        # x = self.block8(x)
        # x = self.block9(x)

        # 第八步，自适应地求一个1*1最大池化
        # ?改进空间
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # 第九步 x展开成长条
        x = x.view(x.size(0), -1)

        # 第十步，最终该分支就会输出对应的特征
        x = self.fea_compress(x)
        # x = self.bn5(x)
        return x

    def forward(self, input):
        # step 1 turn input to level1 DWT high frequency coefficients
        x = input
        fea = self.features(x)

        return fea


class ProcessNet_wave_224_V2(nn.Module):
    def __init__(self, waves_n):
        super().__init__()

        # the argument I need
        in_channel = waves_n * 9
        self.in_channel = in_channel
        # the module I need

        # Head
        self.DWT = WaveGen()
        # self.entry_l1 = nn.Conv2d(
        #     in_channel, in_channel*3, 3, 1, 1, bias=False)

        # self.relu = nn.ReLU(inplace=True)

        # middle flow
        # Block缩小尺寸的方法是池化
        self.bn1 = nn.BatchNorm2d(27)
        self.block1 = Block(
            27, 54, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c1 = Block(
            54, 40, 4, 1, start_with_relu=False, grow_first=True)

        self.bn2 = nn.BatchNorm2d(108)

        self.block2 = Block(
            148, 296, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c2 = Block(
            296, 222, 4, 1, start_with_relu=False, grow_first=True)

        # 此时需要concat l2的DWT
        self.bn3 = nn.BatchNorm2d(432)

        self.block3 = Block(
            654, 1512, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c3 = Block(
            1512, 981, 4, 1, start_with_relu=False, grow_first=True)

        self.block4 = Block(
            981, 1962, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c4 = Block(
            1962, 981, 4, 1, start_with_relu=False, grow_first=True)

        self.block_c5 = Block(
            981, 490, 4, 1, start_with_relu=False, grow_first=True)
        self.block_c6 = Block(
            490, 240, 4, 1, start_with_relu=False, grow_first=True)
        self.block_c7 = Block(
            240, 100, 4, 1, start_with_relu=False, grow_first=True)
        self.block_c8 = Block(
            100, 50, 4, 1, start_with_relu=False, grow_first=True)

        # 压缩
        # self.block4 = Block(
        #     1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        # self.block5 = Block(
        #     1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        # self.block6 = Block(
        #     1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        # self.block7 = Block(
        #     1512, 1512, 3, 1, start_with_relu=False, grow_first=True)
        # self.block8 = Block(
        #     1512, 1512, 3, 1, start_with_relu=False, grow_first=True)

        # self.block9 = Block(
        #     1512, 3024, 4, 2, start_with_relu=False, grow_first=True)

        # self.block7 = Block(
        #     972, 486, 2, 1, start_with_relu=False, grow_first=True)
        # self.block8 = Block(
        #     486, 243, 2, 1, start_with_relu=False, grow_first=True)
        # self.block9 = Block(
        #     243, 128, 2, 1, start_with_relu=False, grow_first=True)

        # self.block10 = Block(
        #     128, 64, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     64, 32, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     32, 16, 2, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block(
        #     16, 8, 2, 1, start_with_relu=False, grow_first=True)
        # self.compress = nn.Sequential(
        #     Block(
        #         972, 486, 2, 1, start_with_relu=False, grow_first=True),
        #     # Block(
        #     #     486, 243, 2, 1, start_with_relu=False, grow_first=True),

        # )

        self.fea_compress = nn.Linear(5000, 1000)
        self.bn5 = nn.BatchNorm2d(1000)

    def features(self, input):
        x = input
        # 第一步 求一阶高频
        x_L, x_H = self.DWT(x)
        x = x_H
        x = self.bn1(x)

        # 第二步 对一阶段的高频系数进行卷积操作，
        x = self.block1(x)
        x = self.block_c1(x)
        # x = self.block2(x)

        # 第三步 求x的二阶段的高频系数
        x_LL, x_LH = self.DWT(x_L)
        x_HL, x_HH = self.DWT(x_H)
        x_t = torch.cat((x_LH, x_HH), dim=1)
        x_t = self.bn2(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)
        x = self.block2(x)
        x = self.block_c2(x)

        # 第四部 对二阶高频做卷积
        # x = self.block3(x)
        # x = self.block4(x)

        # 第五步 求x的第三阶高频系数
        x_LLL, x_LLH = self.DWT(x_LL)
        x_LHL, x_LHH = self.DWT(x_LH)
        x_HLL, x_HLH = self.DWT(x_HL)
        x_HHL, x_HHH = self.DWT(x_HH)
        x_t = torch.cat((x_LLH, x_LHH, x_HLH, x_HHH), dim=1)
        x_t = self.bn3(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # 第六步 对三阶高频做卷积

        x = self.block3(x)
        x = self.block_c3(x)
        x = self.block4(x)
        x = self.block_c4(x)
        x = self.block_c5(x)
        x = self.block_c6(x)
        x = self.block_c7(x)
        x = self.block_c8(x)

        # x = self.bn4(x)

        # 第七步 开始通道压缩,
        # x = self.compress(x)
        # x = self.block6(x)
        # x = self.block7(x)

        # x = self.block8(x)
        # x = self.block9(x)

        # 第八步，自适应地求一个1*1最大池化
        # ?改进空间
        # x = F.adaptive_avg_pool2d(x, (1, 1))

        # 第九步 x展开成长条
        x = x.view(x.size(0), -1)

        # 第十步，最终该分支就会输出对应的特征
        x = self.fea_compress(x)
        x = self.bn5(x)
        return x

    def forward(self, input):
        # step 1 turn input to level1 DWT high frequency coefficients
        x = input
        fea = self.features(x)

        return fea


class ProcessNet_wave_224_V3(nn.Module):
    def __init__(self, waves_n):
        super().__init__()

        self.DWT = WaveGen()

        self.bn1 = nn.BatchNorm2d(27)

        self.block1 = Block(
            27, 54, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c1 = Block(
            54, 27, 4, 1, start_with_relu=False, grow_first=True)

        self.bn2 = nn.BatchNorm2d(108)

        self.block2 = Block(
            135, 270, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c2 = Block(
            270, 135, 4, 1, start_with_relu=False, grow_first=True)

        self.bn3 = nn.BatchNorm2d(432)

        self.block3 = Block(
            567, 1134, 4, 2, start_with_relu=False, grow_first=True)
        self.block_c3 = Block(
            1134, 567, 4, 1, start_with_relu=False, grow_first=True)
        self.block4 = Block(
            567, 1134, 4, 1, start_with_relu=False, grow_first=True)
        self.block_c4 = Block(
            1134, 1134, 4, 1, start_with_relu=False, grow_first=True)

        self.fea_compress = nn.Linear(1134, 1000)
        self.bn5 = nn.BatchNorm2d(1000)

    def features(self, input):
        x = input

        # lv-1 WPT
        x_L, x_H = self.DWT(x)
        x = x_H
        x = self.bn1(x)

        # conv on lv-1 WPT(High)
        x = self.block1(x)
        x = self.block_c1(x)

        # lv-2 WPT
        x_LL, x_LH = self.DWT(x_L)
        x_HL, x_HH = self.DWT(x_H)
        x_t = torch.cat((x_LH, x_HH), dim=1)
        x_t = self.bn2(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # conv on lv-2 WPT(High)+conv(lv-1 WPT(High))
        x = self.block2(x)
        x = self.block_c2(x)

        # lv-3 WPT
        x_LLL, x_LLH = self.DWT(x_LL)
        x_LHL, x_LHH = self.DWT(x_LH)
        x_HLL, x_HLH = self.DWT(x_HL)
        x_HHL, x_HHH = self.DWT(x_HH)
        x_t = torch.cat((x_LLH, x_LHH, x_HLH, x_HHH), dim=1)
        x_t = self.bn3(x_t)
        size = min(x_t.shape[2], x.shape[2])
        x = x[:, :, :size, :size]
        x_t = x_t[:, :, :size, :size]
        x = torch.cat((x_t, x), dim=1)

        # conv on lv-3 WPT(High)+conv(lv-2 WPT(High)+conv(lv-1 WPT(High)))
        x = self.block3(x)
        x = self.block_c3(x)
        x = self.block4(x)
        x = self.block_c4(x)

        # avg pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # flatten
        x = x.view(x.size(0), -1)

        # compress dimension of frequency feature to 1000
        x = self.fea_compress(x)
        x = self.bn5(x)
        return x

    def forward(self, input):

        x = input
        fea = self.features(x)

        return fea


class JudgeNet(nn.Module):
    def __init__(self, fea_n, judgeWay):
        super().__init__()
        self.fea_n = fea_n
        self.judgeWay = judgeWay
        self.j_linear = self.GenJudgeList(fea_n, judgeWay)

        self.sig = nn.Sigmoid()

    def GenJudgeList(self, fea_n, judgeWay):

        return nn.Linear(fea_n, judgeWay)

    def Confidentiate(self, judges):
        x = judges

        judges = torch.mul(x, x)
        sig = torch.sign(x)
        judges = torch.mul(sig, judges)
        return judges

    def forward(self, fea):
        x = fea
        judges = self.j_linear(x)
        judges = self.Confidentiate(judges)
        # judges = torch.Tensor(judges)
        # judges = judges/10

        # 1 先平均值再sigmoid
        # judges = self.sig(judges)
        # judge = judges.mean(1, keepdim=True)

        # 2 先sigmoid再平均值
        judge = judges.mean(1, keepdim=True)

        w = self.j_linear.weight
        return judge, w


class JudgeNet_v2(nn.Module):
    def __init__(self, fea_n, judgeWay):
        super().__init__()
        self.fea_n = fea_n
        self.judgeWay = judgeWay
        self.j_linear = self.GenJudgeList(fea_n, judgeWay)

        self.sig = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(10)
        # self.j_linear.register_backward_hook(self.backward_hook1)

    def GenJudgeList(self, fea_n, judgeWay):

        return nn.Linear(fea_n, judgeWay)

    def Confidentiate(self, judges):

        x = judges

        # for training
        # judges = judges*10

        # for transfer
        # !need to be adjust
        # max_p = torch.max(abs(x[:, 0:5]), dim=1, keepdim=True)[0]
        # max_n = torch.max(abs(x[:, 5:10]), dim=1, keepdim=True)[0]
        # x_p = x[:, 0:5]/max_p
        # x_n = x[:, 5:10]/max_n
        # x = torch.cat((x_p, x_n), dim=1)
        # x = x*2
        x = x/torch.median(abs(x))
        median = 1/torch.median(abs(x), dim=1, keepdim=True)[0]
        # x = x/10
        x = torch.mul(x, median)
        judges = torch.mul(x, x)
        # judges = torch.mul(x, judges)
        sig = torch.sign(x)
        judges = torch.mul(sig, judges)
        # judges = torch.clamp(judges, min=0)
        return judges

    def forward(self, fea):

        x = fea

        # generate confidence score
        judges = self.j_linear(x)
        # judges = self.bn(judges)
        # confidence enhance
        judges = self.Confidentiate(judges)

        # use confidence to generate final scores
        # judges = self.sig(judges)

        # judges[0:4] are positive scores, judges[5:9] are negative scores
        # j_p = torch.topk(judges[:, 0:5], 2, 1)[0]
        # j_n = torch.topk(judges[:, 5:10], 2, 1)[0]

        j_p = torch.mean(judges[:, 0:5], dim=1, keepdim=True)
        j_n = torch.mean(judges[:, 5:10], dim=1, keepdim=True)
        # j_p = torch.mean(j_p, dim=1, keepdim=True)
        # j_n = torch.mean(j_n, dim=1, keepdim=True)

        j_p = self.sig(j_p)
        j_n = self.sig(j_n)

        # j_p = torch.clamp(self.sig(j_p)+0.2, max=1)
        # j_n = torch.clamp(self.sig(j_n)+0.2, max=1)

        # W_class for L_FA, L_FB
        w = self.j_linear.weight

        judge = torch.cat((j_p, j_n), 1)

        return judge, w


class feaProcess(nn.Module):
    def __init__(self, fea_n):
        super().__init__()
        self.feaCompress1 = nn.Linear(fea_n, 1000)
        self.dp1 = nn.Dropout(p=0.2)
        self.act1 = nn.LeakyReLU(0.01)
        self.dp2 = nn.Dropout(p=0.2)

    def features(self, input):
        x = input
        x = self.feaCompress1(x)
        x = self.dp1(x)

        return x

    def forward(self, input):

        # use linear turn
        x = self.features(input)
        w_1 = self.feaCompress1.weight

        return x, w_1


# the whole net
class WholeNet(nn.Module):
    def __init__(self, t_path):
        super().__init__()
        # self.branch1 = self.EfficientNetB4Gen()
        self.branch1 = self.XceptionGen(t_path)
        self.branch2 = ProcessNet_wave_224_V3(3)
        self.feaProcess = feaProcess(3048)
        self.judge = JudgeNet_v2(1000, 10)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_b1_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.branch1.parameters())
        return a

    def get_b2_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.branch2.parameters())
        return a

    def get_fp_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.feaProcess.parameters())
        return a

    def get_j_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.judge.parameters())
        return a

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def XceptionGen(self, t_path):
        print("Loading weight for spatial branch")
        path = t_path
        net = xception()
        state_dict = torch.load(path)
        incomp_keys = net.load_state_dict(
            {k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        print(incomp_keys)

        return net

    def featureCon(self, fea_l):
        return torch.cat((fea_l), dim=0)

    def forward(self, input):
        x = input

        # step 1 dispatch data to branches,and then they shall return the features we need for judging
        fea1 = self.branch1.features(x)
        fea2 = self.branch2(x)

        # step 2 feature concat and pass feature to feature process net
        fea = torch.cat((fea1, fea2), dim=1)
        fea, w_feaP = self.feaProcess(fea)

        # step 3 use processed feature to make judgement
        judge, w_j = self.judge(fea)

        # step 4 collect weight form judge net and  feature process net
        w = torch.mm(w_j, w_feaP)
        w = torch.abs(w)

        return judge, w


class WholeNet_forAlabation(nn.Module):
    def __init__(self):
        super().__init__()
        # self.branch1 = self.EfficientNetB4Gen()
        self.branch1 = self.XceptionGen()
        self.branch2 = ProcessNet_wave_224_V3(3)
        # self.fc = nn.Linear(3048, 1)
        self.feaProcess = feaProcess(3048)
        self.judge = JudgeNet(1000, 10)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    def get_b1_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.branch1.parameters())
        return a

    def get_b2_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.branch2.parameters())
        return a

    def get_fp_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.feaProcess.parameters())
        return a

    def get_j_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.judge.parameters())
        return a

    def get_fc_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.fc.parameters())
        return a

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def XceptionGen(self):
        print("Loading weight for spatial branch")
        path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-c23_FS/bestval.pth"
        net = xception()
        # state_dict = torch.load(path)
        # incomp_keys = net.load_state_dict(
        #     {k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        # print(incomp_keys)
        # for name, parameter in net.named_parameters():
        #     parameter.requries_grad = False
        return net

    def featureCon(self, fea_l):
        return torch.cat((fea_l), dim=0)

    def forward(self, input):
        x = input
        # step 1 dispatch data to branches,and then they shall return the features we need for judging
        fea1 = self.branch1.features(x)
        fea2 = self.branch2(x)
        # 用于运算KD_loss
        # t_out = self.branch1.classifier(fea1)
        # t_out = torch.sigmoid(t_out)
        fea = torch.cat((fea1, fea2), dim=1)
        # fea = fea2

        fea, w_feaP = self.feaProcess(fea)
        judge, w_j = self.judge(fea)
        w = torch.mm(w_j, w_feaP)

        # judge = self.fc(fea)
        # w = torch.abs(w)
        # pow_fea1 = w[0:1396].mean()
        # pow_fea2 = w[1396:].mean()
        return judge, w


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        fea = self.features(x)

        return x


class xception(Xception):
    """
    Construct Xception.
    """

    def __init__(self):
        super(Xception, self).__init__()
        self.Xce = Xception()

        # self.Xce.load_state_dict(
        #     model_zoo.load_url(model_urls['xception']))

        self.classifier = nn.Linear(2048, 2)

    def features(self, x):
        x = self.Xce.features(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_parameters(self):
        return self.Xce.parameters()

    def get_normalizer(self):
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class MesoInception4(nn.Module):

    def __init__(self):

        super(MesoInception4, self).__init__()

        self.inceptionL1 = InceptionLayer(3, 1, 4, 4, 2)
        self.bn1 = nn.BatchNorm2d(11)
        self.maxp1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.inceptionL2 = InceptionLayer(11, 2, 4, 4, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv1 = nn.Conv2d(12, 16, 5, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxp3 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)

        self.conv2 = nn.Conv2d(16, 16, 5, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(16)
        self.maxp4 = nn.MaxPool2d(kernel_size=(4, 4), padding=2)

        self.dense = nn.Linear(1600, 16)
        self.leakyRelu = nn.LeakyReLU(0.1, inplace=True)
        self.classifier = nn.Linear(16, 2)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def features(self, x):

        x = self.inceptionL1(x)
        x = self.bn1(x)
        x = self.maxp1(x)

        x = self.inceptionL2(x)
        x = self.bn2(x)
        x = self.maxp2(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn3(x)
        x = self.maxp3(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn4(x)
        x = self.maxp4(x)

        return x

    def logits(self, x):
        x = x.view(x.size(0), -1)
        x = nn.Dropout(0.5)(x)
        x = self.dense(x)
        x = self.leakyRelu(x)
        x = nn.Dropout(0.5)(x)
        x = self.classifier(x)

        return x

    def forward(self, x):
        fea = self.features(x)
        x = self.logits(fea)
        return x


class InceptionLayer(nn.Module):
    def __init__(self, input_c, a, b, c, d):

        super(InceptionLayer, self).__init__()

        self.l1 = nn.Conv2d(input_c, a, 1, 1, 0)

        self.l2 = nn.Conv2d(input_c, b, 1, 1, 0)
        self.l3 = nn.Conv2d(b, b, 3, 1, 1)

        self.l4 = nn.Conv2d(input_c, c, 1, 1, 0)
        self.l5 = nn.Conv2d(c, c, 3, 1, 2, 2)

        self.l6 = nn.Conv2d(input_c, d, 1, 1, 0)
        self.l7 = nn.Conv2d(d, d, 3, 1, 3, 3)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x1 = self.l1(x)

        x2 = self.l2(x)
        x2 = self.l3(x2)

        x3 = self.l4(x)
        x3 = self.l5(x3)

        x4 = self.l6(x)
        x4 = self.l7(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()
        # 载入预处理模型
        self.efficientnet = EfficientNet.from_pretrained(model)
        # 将模型最后输出转化为唯一的logit
        self.classifier = nn.Linear(
            self.efficientnet._conv_head.out_channels, 2)
        del self.efficientnet._fc

    # 模型将输入转化为logits的整个过程
    def features(self, x: torch.Tensor) -> torch.Tensor:
        # 先用预训练的EfficientNet进行特征提取
        x = self.efficientnet.extract_features(x)
        # 对特征信息进行平均池化
        x = self.efficientnet._avg_pooling(x)
        # 拉成一根长条
        x = x.flatten(start_dim=1)
        return x

    # 将输出进行处理后
    def forward(self, x):
        x = self.features(x)
        # 对一部划分的数据进行dropout
        x = self.efficientnet._dropout(x)
        ##
        x = self.classifier(x)
        return x


class EfficientNetB4(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4')
