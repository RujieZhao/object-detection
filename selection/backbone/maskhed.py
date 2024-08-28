import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_int
from detectron2.layers import Conv2d
from torchvision.models import vgg16


class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=21)
        # self.norm1 = nn.InstanceNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        # self.norm2 = nn.InstanceNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        # self.norm3 = nn.InstanceNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.norm4 = nn.InstanceNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        # self.norm5 = nn.InstanceNorm2d(512)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x,img_H,img_W):
        # VGG
        # img_H, img_W = x.shape[2], x.shape[3]
        # conv1_1 = self.relu(self.norm1(self.conv1_1(x)))
        conv1_1 = self.relu(self.conv1_1(x))
        # conv1_2 = self.relu(self.norm1(self.conv1_2(conv1_1)))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        # conv2_1 = self.relu(self.norm2(self.conv2_1(pool1)))
        conv2_1 = self.relu(self.conv2_1(pool1))
        # conv2_2 = self.relu(self.norm2(self.conv2_2(conv2_1)))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        # conv3_1 = self.relu(self.norm3(self.conv3_1(pool2)))
        conv3_1 = self.relu(self.conv3_1(pool2))
        # conv3_2 = self.relu(self.norm3(self.conv3_2(conv3_1)))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        # conv3_3 = self.relu(self.norm3(self.conv3_3(conv3_2)))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        # conv4_1 = self.relu(self.norm4(self.conv4_1(pool3)))
        conv4_1 = self.relu(self.conv4_1(pool3))
        # conv4_2 = self.relu(self.norm4(self.conv4_2(conv4_1)))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        # conv4_3 = self.relu(self.norm4(self.conv4_3(conv4_2)))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool(conv4_3)

        # conv5_1 = self.relu(self.norm5(self.conv5_1(pool4)))
        conv5_1 = self.relu(self.conv5_1(pool4))
        # conv5_2 = self.relu(self.norm5(self.conv5_2(conv5_1)))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        # conv5_3 = self.relu(self.norm5(self.conv5_3(conv5_2)))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        so1 = self.score_dsn1(conv1_2)
        so2 = self.score_dsn2(conv2_2)
        so3 = self.score_dsn3(conv3_3)
        so4 = self.score_dsn4(conv4_3)
        so5 = self.score_dsn5(conv5_3)
        print(so1.shape, so2.shape, so3.shape, so4.shape, so5.shape)
        weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        weight_deconv5 = make_bilinear_weights(32, 1).cuda()

        upsample2 = torch.nn.functional.conv_transpose2d(so2, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5, weight_deconv5, stride=16)

        so1 = crop(so1, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        print(so1.shape,so2.shape,so3.shape,so4.shape,so5.shape)
        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        # results = [so1, so2, so3, so4, so5, fuse]
        # results = [torch.sigmoid(r) for r in results]
        return fuse

def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]

# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(in_channels, out_channels, h, w):
    weights = np.zeros([in_channels, out_channels, h, w])
    if in_channels != out_channels:
        raise ValueError("Input Output channel!")
    if h != w:
        raise ValueError("filters need to be square!")
    filt = upsample_filt(h)
    weights[range(in_channels), range(out_channels), :, :] = filt
    return np.float32(weights)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)



class rujiemaskbb(nn.Module):
    def __init__(self,inch):
        super(rujiemaskbb,self).__init__()
        # self.vgg_new = decom_vgg16() #<class 'torch.nn.modules.container.Sequential'>
        # self.vgg_new = vgg16().features[:15]
        # '''

        self.norm = nn.InstanceNorm2d(inch)
        self.act = nn.PReLU()
        self.conv = nn.Sequential(
            (Conv2d(inch,inch,kernel_size = 3,stride = 1,padding = 1,norm = self.norm, activation = self.act)),
            (Conv2d(inch, inch, kernel_size=3, stride=1, padding=1, norm=self.norm, activation=self.act)),
            (Conv2d(inch, inch, kernel_size=3, stride=1, padding=1, norm=self.norm, activation=self.act)),
            (Conv2d(inch, inch, kernel_size=3, stride=1, padding=1, norm=self.norm, activation=self.act)),
        )
        self.vgg_zrj = vgg16().features[:15]
        self.last_mask = nn.Sequential(
            (Conv2d((inch // 16), (inch // 32), kernel_size=(1, 1), stride=(1, 1),norm = self.norm, activation = self.act)),
            (Conv2d((inch // 32), 1, kernel_size=(1, 1), stride=(1, 1))))

        for i in [self.conv,self.last_mask]:
        # for i in self.conv:
            if isinstance(i,nn.Conv2d):
                # print("seprate:",i)
                nn.init.normal_(i.weight,std=0.001)
                nn.init.constant_(i.bias,0)
                # weight_int.c2_msra_fill(i)
        # for i in self.last_mask:
        # 	if isinstance(i,nn.Conv2d):
        # 		nn.init.normal_(i.weight,std=0.001)
        # 		nn.init.constant_(i.bias,0)
                # weight_int.c2_msra_fill(i)
    def lossmaker(self,pred,gt,gamma=2.,alpha=0.25):
        gt = gt.long()
        mask = (gt != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        mask[gt != 0] = num_negative / (gt.numel())
        mask[gt == 0] = num_positive / (gt.numel()) #*5

        prob = pred.sigmoid()
        # ce_loss = F.binary_cross_entropy(pred,gt,reduction="none")
        ce_loss = F.binary_cross_entropy(pred.float(),gt.float(),weight=mask,reduction="none")
        # return ce_loss.mean()
        p_t = prob*gt+(1-prob)*(1-gt)
        loss = ce_loss*((1-p_t)**gamma)
        # if alpha >=0:
        # 	alpha_t = alpha*gt+(1-alpha)*(1-gt)
        # 	loss = alpha_t*loss
        return loss.sum()

    def forward(self,x,img_h,img_w):
        output = self.vgg_zrj(x)
        output = self.act(self.norm(output))
        # output = self.conv(x)
        output = self.conv(output)
        output = output.view(1, 16, output.shape[-2] * 4, output.shape[-1] * 4)
        if output.shape[-2] * 4 != img_h and output.shape[-1] * 4 != img_w:
            output = F.interpolate(output, size=[img_h, img_w], mode="nearest")
        # output = self.act(output)
        output = self.last_mask(output) #.sigmoid()
        # output = self.last_mask(output)
        return output





# if __name__ == "__main__":
#     device = torch.device("cuda:0") if torch.cuda.is_available else "cpu"
#     input  = torch.rand((1,3, 640, 960)).to(device)
#     model = HED().to(device) #rujiemaskbb(256)
#     output = model(input,640,959)
#     print("outputshape:",output.shape)




