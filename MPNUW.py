import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import GoProDataset
import time

from collections import OrderedDict

# from torch.nn.modules.loss import _Loss

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=25)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=4)
parser.add_argument("-s", "--imagesize", type=int, default=256)
parser.add_argument("-l", "--learning_rate", type=float, default=0.000001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "124816MPN"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize


class UnderwaterLoss1(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, output, sharp, ):
        perloss = torch.norm(output - sharp)

        loss = perloss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnderwaterLoss2(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, output, sharp):
        edgeloss = torch.mean(torch.sqrt((output - sharp) ** 2 + 1e-3 ** 2))

        # conloss = torch.sum((output - sharp) ** 2) / (512 * 512 * 3)

        loss = 0.5 * edgeloss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def save_deblur_images(images, iteration, epoch, save_every=1):
    # 仅在特定 epoch（如每隔 save_every 个）保存
    if epoch % save_every == 0:
        folder_path = f'./checkpoints/{METHOD}/epoch{epoch}'

        # 检查并创建目录
        os.makedirs(folder_path, exist_ok=True)

        filename = os.path.join(folder_path, f"Iter_{iteration}_deblur.png")

        # 保存图像
        torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    print("init data folders")

    encoder_lv1 = models.Encoder()
    encoder_lv2_a = models.Encoder()
    encoder_lv2_b = models.Encoder()
    encoder_lv3 = models.Encoder()
    encoder_lv4_a = models.Encoder()
    encoder_lv4_b = models.Encoder()
    encoder_lv5 = models.Encoder()

    encoder_lv1 = nn.DataParallel(encoder_lv1)
    encoder_lv2_a = nn.DataParallel(encoder_lv2_a)
    encoder_lv2_b = nn.DataParallel(encoder_lv2_b)
    encoder_lv3 = nn.DataParallel(encoder_lv3)
    encoder_lv4_a = nn.DataParallel(encoder_lv4_a)
    encoder_lv4_b = nn.DataParallel(encoder_lv4_b)
    encoder_lv5 = nn.DataParallel(encoder_lv5)

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()
    decoder_lv3_a = models.Decoder()
    decoder_lv3_b = models.Decoder()
    decoder_lv4 = models.Decoder()
    decoder_lv5_a = models.Decoder()
    decoder_lv5_b = models.Decoder()

    decoder_lv1 = nn.DataParallel(decoder_lv1)
    decoder_lv2 = nn.DataParallel(decoder_lv2)
    decoder_lv3_a = nn.DataParallel(decoder_lv3_a)
    decoder_lv3_b = nn.DataParallel(decoder_lv3_b)
    decoder_lv4 = nn.DataParallel(decoder_lv4)
    decoder_lv5_a = nn.DataParallel(decoder_lv5_a)
    decoder_lv5_b = nn.DataParallel(decoder_lv5_b)

    encoder_lv1.apply(weight_init).cuda(GPU)
    encoder_lv2_a.apply(weight_init).cuda(GPU)
    encoder_lv2_b.apply(weight_init).cuda(GPU)
    encoder_lv3.apply(weight_init).cuda(GPU)
    encoder_lv4_a.apply(weight_init).cuda(GPU)
    encoder_lv4_b.apply(weight_init).cuda(GPU)
    encoder_lv5.apply(weight_init).cuda(GPU)

    decoder_lv1.apply(weight_init).cuda(GPU)
    decoder_lv2.apply(weight_init).cuda(GPU)
    decoder_lv3_a.apply(weight_init).cuda(GPU)
    decoder_lv3_b.apply(weight_init).cuda(GPU)
    decoder_lv4.apply(weight_init).cuda(GPU)
    decoder_lv5_a.apply(weight_init).cuda(GPU)
    decoder_lv5_b.apply(weight_init).cuda(GPU)

    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(), lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(encoder_lv1_optim, step_size=1000, gamma=0.1)

    encoder_lv2_a_optim = torch.optim.Adam(encoder_lv2_a.parameters(), lr=LEARNING_RATE)
    encoder_lv2_a_scheduler = StepLR(encoder_lv2_a_optim, step_size=1000, gamma=0.1)
    encoder_lv2_b_optim = torch.optim.Adam(encoder_lv2_b.parameters(), lr=LEARNING_RATE)
    encoder_lv2_b_scheduler = StepLR(encoder_lv2_b_optim, step_size=1000, gamma=0.1)

    encoder_lv3_optim = torch.optim.Adam(encoder_lv3.parameters(), lr=LEARNING_RATE)
    encoder_lv3_scheduler = StepLR(encoder_lv3_optim, step_size=1000, gamma=0.1)

    encoder_lv4_a_optim = torch.optim.Adam(encoder_lv4_a.parameters(), lr=LEARNING_RATE)
    encoder_lv4_a_scheduler = StepLR(encoder_lv4_a_optim, step_size=1000, gamma=0.1)
    encoder_lv4_b_optim = torch.optim.Adam(encoder_lv4_b.parameters(), lr=LEARNING_RATE)
    encoder_lv4_b_scheduler = StepLR(encoder_lv4_b_optim, step_size=1000, gamma=0.1)

    encoder_lv5_optim = torch.optim.Adam(encoder_lv5.parameters(), lr=LEARNING_RATE)
    encoder_lv5_scheduler = StepLR(encoder_lv5_optim, step_size=1000, gamma=0.1)
    # ------------------------------------------------------------------------------------------------
    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(), lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(decoder_lv1_optim, step_size=1000, gamma=0.1)

    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(), lr=LEARNING_RATE)
    decoder_lv2_scheduler = StepLR(decoder_lv2_optim, step_size=1000, gamma=0.1)

    decoder_lv3_a_optim = torch.optim.Adam(decoder_lv3_a.parameters(), lr=LEARNING_RATE)
    decoder_lv3_a_scheduler = StepLR(decoder_lv3_a_optim, step_size=1000, gamma=0.1)
    decoder_lv3_b_optim = torch.optim.Adam(decoder_lv3_b.parameters(), lr=LEARNING_RATE)
    decoder_lv3_b_scheduler = StepLR(decoder_lv3_b_optim, step_size=1000, gamma=0.1)

    decoder_lv4_optim = torch.optim.Adam(decoder_lv4.parameters(), lr=LEARNING_RATE)
    decoder_lv4_scheduler = StepLR(decoder_lv4_optim, step_size=1000, gamma=0.1)

    decoder_lv5_a_optim = torch.optim.Adam(decoder_lv5_a.parameters(), lr=LEARNING_RATE)
    decoder_lv5_a_scheduler = StepLR(decoder_lv5_a_optim, step_size=1000, gamma=0.1)
    decoder_lv5_b_optim = torch.optim.Adam(decoder_lv5_b.parameters(), lr=LEARNING_RATE)
    decoder_lv5_b_scheduler = StepLR(decoder_lv5_b_optim, step_size=1000, gamma=0.1)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2_a.pkl")):
        encoder_lv2_a.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2_a.pkl")))
        print("load encoder_lv2_a success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2_b.pkl")):
        encoder_lv2_b.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2_b.pkl")))
        print("load encoder_lv2_b success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv4_a.pkl")):
        encoder_lv4_a.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv4_a.pkl")))
        print("load encoder_lv4_a success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv4_b.pkl")):
        encoder_lv4_b.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv4_b.pkl")))
        print("load encoder_lv4_b success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv5.pkl")):
        encoder_lv5.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv5.pkl")))
        print("load encoder_lv5 success")

    # ------
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load decoder_lv1 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3_a.pkl")):
        decoder_lv3_a.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3_a.pkl")))
        print("load decoder_lv3_a success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3_b.pkl")):
        decoder_lv3_b.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3_b.pkl")))
        print("load decoder_lv3_b success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")):
        decoder_lv4.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv4.pkl")))
        print("load decoder_lv4 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv5_a.pkl")):
        decoder_lv5_a.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv5_a.pkl")))
        print("load decoder_lv5_a success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv5_b.pkl")):
        decoder_lv5_b.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv5_b.pkl")))
        print("load decoder_lv5_b success")

    if not os.path.exists('./checkpoints/' + METHOD):
        os.system('mkdir ./checkpoints/' + METHOD)

    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_a_scheduler.step(epoch)
        encoder_lv2_b_scheduler.step(epoch)
        encoder_lv3_scheduler.step(epoch)
        encoder_lv4_a_scheduler.step(epoch)
        encoder_lv4_b_scheduler.step(epoch)
        encoder_lv5_scheduler.step(epoch)
        # -------
        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)
        decoder_lv3_a_scheduler.step(epoch)
        decoder_lv3_b_scheduler.step(epoch)
        decoder_lv4_scheduler.step(epoch)
        decoder_lv5_a_scheduler.step(epoch)
        decoder_lv5_b_scheduler.step(epoch)

        print("Training...")

        train_dataset = GoProDataset(
            blur_image_files='./datas/UCDDA/train_blur_file.txt',
            sharp_image_files='./datas/UCDDA/train_sharp_file.txt',
            root_dir='./datas/UCDDA',
            crop=True,
            crop_size=IMAGE_SIZE,
            transform=transforms.Compose([transforms.ToTensor()]))
        # train_dataset = GoProDataset(
        #     blur_image_files='./datas/GoPro/train_blur_file.txt',
        #     sharp_image_files='./datas/GoPro/train_sharp_file.txt',
        #     root_dir='./datas/GoPro',
        #     crop=True,
        #     crop_size=IMAGE_SIZE,
        #     transform=transforms.Compose([transforms.ToTensor()]))
        # train_dataset = GoProDataset(
        #     blur_image_files='./datas/UCDD_originsize/train_blur_file.txt',
        #     sharp_image_files='./datas/UCDD_originsize/train_sharp_file.txt',
        #     root_dir='./datas/UCDD_originsize',
        #     crop=True,
        #     crop_size=IMAGE_SIZE,
        #     transform=transforms.Compose([transforms.ToTensor()]))
        # train_dataset = GoProDataset(
        #     blur_image_files='./datas/UCDD_1280_II/train_blur_file.txt',
        #     sharp_image_files='./datas/UCDD_1280_II/train_sharp_file.txt',
        #     root_dir='./datas/UCDD_1280_II',
        #     crop=True,
        #     crop_size=IMAGE_SIZE,
        #     transform=transforms.Compose([transforms.ToTensor()]))

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        start = 0
        # sum_loss = 0

        for iteration, images in enumerate(train_dataloader):
            mse = nn.MSELoss().cuda(GPU)
            # per =

            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)

            H = gt.size(2)
            W = gt.size(3)

            gt_lv1 = Variable(images['sharp_image'] - 0.5).cuda(GPU)
            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)

            images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
            images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

            images_lv2_3 = images_lv1[:, :, :, 0:int(W / 2)]
            images_lv2_4 = images_lv1[:, :, :, int(W / 2):W]

            images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
            images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
            images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
            images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]

            images_lv4_1 = images_lv3_1[:, :, 0:int(H / 4), :]
            images_lv4_2 = images_lv3_1[:, :, int(H / 4):int(H / 2), :]
            images_lv4_3 = images_lv3_2[:, :, 0:int(H / 4), :]
            images_lv4_4 = images_lv3_2[:, :, int(H / 4):int(H / 2), :]
            images_lv4_5 = images_lv3_3[:, :, 0:int(H / 4), :]
            images_lv4_6 = images_lv3_3[:, :, int(H / 4):int(H / 2), :]
            images_lv4_7 = images_lv3_4[:, :, 0:int(H / 4), :]
            images_lv4_8 = images_lv3_4[:, :, int(H / 4):int(H / 2), :]

            images_lv4_9 = images_lv3_1[:, :, :, 0:int(W / 4)]
            images_lv4_10 = images_lv3_1[:, :, :, int(W / 4):int(W / 2)]
            images_lv4_11 = images_lv3_2[:, :, :, 0:int(W / 4)]
            images_lv4_12 = images_lv3_2[:, :, :, int(W / 4):int(W / 2)]
            images_lv4_13 = images_lv3_3[:, :, :, 0:int(W / 4)]
            images_lv4_14 = images_lv3_3[:, :, :, int(W / 4):int(W / 2)]
            images_lv4_15 = images_lv3_4[:, :, :, 0:int(W / 4)]
            images_lv4_16 = images_lv3_4[:, :, :, int(W / 4):int(W / 2)]

            images_lv5_1 = images_lv4_1[:, :, :, 0:int(W / 4)]
            images_lv5_2 = images_lv4_1[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_3 = images_lv4_2[:, :, :, 0:int(W / 4)]
            images_lv5_4 = images_lv4_2[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_5 = images_lv4_3[:, :, :, 0:int(W / 4)]
            images_lv5_6 = images_lv4_3[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_7 = images_lv4_4[:, :, :, 0:int(W / 4)]
            images_lv5_8 = images_lv4_4[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_9 = images_lv4_5[:, :, :, 0:int(W / 4)]
            images_lv5_10 = images_lv4_5[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_11 = images_lv4_6[:, :, :, 0:int(W / 4)]
            images_lv5_12 = images_lv4_6[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_13 = images_lv4_7[:, :, :, 0:int(W / 4)]
            images_lv5_14 = images_lv4_7[:, :, :, int(W / 4):int(W / 2)]
            images_lv5_15 = images_lv4_8[:, :, :, 0:int(W / 4)]
            images_lv5_16 = images_lv4_8[:, :, :, int(W / 4):int(W / 2)]

            # ---------------------------------------------------------

            gt_lv2_1 = gt_lv1[:, :, 0:int(H / 2), :]
            gt_lv2_2 = gt_lv1[:, :, int(H / 2):H, :]

            # gt_lv2_3 = gt_lv1[:, :, :, 0:int(W / 2)]
            # gt_lv2_4 = gt_lv1[:, :, :, int(W / 2):W]

            gt_lv3_1 = gt_lv2_1[:, :, :, 0:int(W / 2)]
            gt_lv3_2 = gt_lv2_1[:, :, :, int(W / 2):W]
            gt_lv3_3 = gt_lv2_2[:, :, :, 0:int(W / 2)]
            gt_lv3_4 = gt_lv2_2[:, :, :, int(W / 2):W]

            gt_lv4_1 = gt_lv3_1[:, :, 0:int(H / 4), :]
            gt_lv4_2 = gt_lv3_1[:, :, int(H / 4):int(H / 2), :]
            gt_lv4_3 = gt_lv3_2[:, :, 0:int(H / 4), :]
            gt_lv4_4 = gt_lv3_2[:, :, int(H / 4):int(H / 2), :]
            gt_lv4_5 = gt_lv3_3[:, :, 0:int(H / 4), :]
            gt_lv4_6 = gt_lv3_3[:, :, int(H / 4):int(H / 2), :]
            gt_lv4_7 = gt_lv3_4[:, :, 0:int(H / 4), :]
            gt_lv4_8 = gt_lv3_4[:, :, int(H / 4):int(H / 2), :]

            # gt_lv4_9 = gt_lv3_1[:, :, :, 0:int(W / 4)]
            # gt_lv4_10 = gt_lv3_1[:, :, :, int(W / 4):int(W / 2)]
            # gt_lv4_11 = gt_lv3_2[:, :, :, 0:int(W / 4)]
            # gt_lv4_12 = gt_lv3_2[:, :, :, int(W / 4):int(W / 2)]
            # gt_lv4_13 = gt_lv3_3[:, :, :, 0:int(W / 4)]
            # gt_lv4_14 = gt_lv3_3[:, :, :, int(W / 4):int(W / 2)]
            # gt_lv4_15 = gt_lv3_4[:, :, :, 0:int(W / 4)]
            # gt_lv4_16 = gt_lv3_4[:, :, :, int(W / 4):int(W / 2)]

            gt_lv5_1 = gt_lv4_1[:, :, :, 0:int(W / 4)]
            gt_lv5_2 = gt_lv4_1[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_3 = gt_lv4_2[:, :, :, 0:int(W / 4)]
            gt_lv5_4 = gt_lv4_2[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_5 = gt_lv4_3[:, :, :, 0:int(W / 4)]
            gt_lv5_6 = gt_lv4_3[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_7 = gt_lv4_4[:, :, :, 0:int(W / 4)]
            gt_lv5_8 = gt_lv4_4[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_9 = gt_lv4_5[:, :, :, 0:int(W / 4)]
            gt_lv5_10 = gt_lv4_5[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_11 = gt_lv4_6[:, :, :, 0:int(W / 4)]
            gt_lv5_12 = gt_lv4_6[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_13 = gt_lv4_7[:, :, :, 0:int(W / 4)]
            gt_lv5_14 = gt_lv4_7[:, :, :, int(W / 4):int(W / 2)]
            gt_lv5_15 = gt_lv4_8[:, :, :, 0:int(W / 4)]
            gt_lv5_16 = gt_lv4_8[:, :, :, int(W / 4):int(W / 2)]

            # ----------
            feature_lv5_1 = encoder_lv5(images_lv5_1)
            feature_lv5_2 = encoder_lv5(images_lv5_2)
            feature_lv5_3 = encoder_lv5(images_lv5_3)
            feature_lv5_4 = encoder_lv5(images_lv5_4)
            feature_lv5_5 = encoder_lv5(images_lv5_5)
            feature_lv5_6 = encoder_lv5(images_lv5_6)
            feature_lv5_7 = encoder_lv5(images_lv5_7)
            feature_lv5_8 = encoder_lv5(images_lv5_8)
            feature_lv5_9 = encoder_lv5(images_lv5_9)
            feature_lv5_10 = encoder_lv5(images_lv5_10)
            feature_lv5_11 = encoder_lv5(images_lv5_11)
            feature_lv5_12 = encoder_lv5(images_lv5_12)
            feature_lv5_13 = encoder_lv5(images_lv5_13)
            feature_lv5_14 = encoder_lv5(images_lv5_14)
            feature_lv5_15 = encoder_lv5(images_lv5_15)
            feature_lv5_16 = encoder_lv5(images_lv5_16)

            # --------------------------------
            gtfeature_lv5_1 = encoder_lv5(gt_lv5_1)
            gtfeature_lv5_2 = encoder_lv5(gt_lv5_2)
            gtfeature_lv5_3 = encoder_lv5(gt_lv5_3)
            gtfeature_lv5_4 = encoder_lv5(gt_lv5_4)
            gtfeature_lv5_5 = encoder_lv5(gt_lv5_5)
            gtfeature_lv5_6 = encoder_lv5(gt_lv5_6)
            gtfeature_lv5_7 = encoder_lv5(gt_lv5_7)
            gtfeature_lv5_8 = encoder_lv5(gt_lv5_8)
            gtfeature_lv5_9 = encoder_lv5(gt_lv5_9)
            gtfeature_lv5_10 = encoder_lv5(gt_lv5_10)
            gtfeature_lv5_11 = encoder_lv5(gt_lv5_11)
            gtfeature_lv5_12 = encoder_lv5(gt_lv5_12)
            gtfeature_lv5_13 = encoder_lv5(gt_lv5_13)
            gtfeature_lv5_14 = encoder_lv5(gt_lv5_14)
            gtfeature_lv5_15 = encoder_lv5(gt_lv5_15)
            gtfeature_lv5_16 = encoder_lv5(gt_lv5_16)

            gtfeature_lv5_top_left_top = torch.cat((gtfeature_lv5_1, gtfeature_lv5_2), 3)
            gtfeature_lv5_top_left_bot = torch.cat((gtfeature_lv5_3, gtfeature_lv5_4), 3)
            gtfeature_lv5_top_right_top = torch.cat((gtfeature_lv5_5, gtfeature_lv5_6), 3)
            gtfeature_lv5_top_right_bot = torch.cat((gtfeature_lv5_7, gtfeature_lv5_8), 3)
            gtfeature_lv5_bot_left_top = torch.cat((gtfeature_lv5_9, gtfeature_lv5_10), 3)
            gtfeature_lv5_bot_left_bot = torch.cat((gtfeature_lv5_11, gtfeature_lv5_12), 3)
            gtfeature_lv5_bot_right_top = torch.cat((gtfeature_lv5_13, gtfeature_lv5_14), 3)
            gtfeature_lv5_bot_right_bot = torch.cat((gtfeature_lv5_15, gtfeature_lv5_16), 3)

            gtfeature_lv5_top_left = torch.cat((gtfeature_lv5_top_left_top, gtfeature_lv5_top_left_bot), 2)
            gtfeature_lv5_top_right = torch.cat((gtfeature_lv5_top_right_top, gtfeature_lv5_top_right_bot), 2)
            gtfeature_lv5_bot_left = torch.cat((gtfeature_lv5_bot_left_top, gtfeature_lv5_bot_left_bot), 2)
            gtfeature_lv5_bot_right = torch.cat((gtfeature_lv5_bot_right_top, gtfeature_lv5_bot_right_bot), 2)

            gtfeature_lv5_top = torch.cat((gtfeature_lv5_top_left, gtfeature_lv5_top_right), 3)
            gtfeature_lv5_bot = torch.cat((gtfeature_lv5_bot_left, gtfeature_lv5_bot_right), 3)

            gtfeature_lv5 = torch.cat((gtfeature_lv5_top, gtfeature_lv5_bot), 2)

            feature_lv5_top_left_top = torch.cat((feature_lv5_1, feature_lv5_2), 3)
            feature_lv5_top_left_bot = torch.cat((feature_lv5_3, feature_lv5_4), 3)
            feature_lv5_top_right_top = torch.cat((feature_lv5_5, feature_lv5_6), 3)
            feature_lv5_top_right_bot = torch.cat((feature_lv5_7, feature_lv5_8), 3)
            feature_lv5_bot_left_top = torch.cat((feature_lv5_9, feature_lv5_10), 3)
            feature_lv5_bot_left_bot = torch.cat((feature_lv5_11, feature_lv5_12), 3)
            feature_lv5_bot_right_top = torch.cat((feature_lv5_13, feature_lv5_14), 3)
            feature_lv5_bot_right_bot = torch.cat((feature_lv5_15, feature_lv5_16), 3)

            feature_lv5_top_left_left = torch.cat((feature_lv5_1, feature_lv5_3), 2)
            feature_lv5_top_left_right = torch.cat((feature_lv5_2, feature_lv5_4), 2)
            feature_lv5_top_right_left = torch.cat((feature_lv5_5, feature_lv5_7), 2)
            feature_lv5_top_right_right = torch.cat((feature_lv5_6, feature_lv5_8), 2)
            feature_lv5_bot_left_left = torch.cat((feature_lv5_9, feature_lv5_11), 2)
            feature_lv5_bot_left_right = torch.cat((feature_lv5_10, feature_lv5_12), 2)
            feature_lv5_bot_right_left = torch.cat((feature_lv5_13, feature_lv5_15), 2)
            feature_lv5_bot_right_right = torch.cat((feature_lv5_14, feature_lv5_16), 2)

            feature_lv5_top_left = torch.cat((feature_lv5_top_left_top, feature_lv5_top_left_bot), 2)
            feature_lv5_top_right = torch.cat((feature_lv5_top_right_top, feature_lv5_top_right_bot), 2)
            feature_lv5_bot_left = torch.cat((feature_lv5_bot_left_top, feature_lv5_bot_left_bot), 2)
            feature_lv5_bot_right = torch.cat((feature_lv5_bot_right_top, feature_lv5_bot_right_bot), 2)

            feature_lv5_top = torch.cat((feature_lv5_top_left, feature_lv5_top_right), 3)
            feature_lv5_bot = torch.cat((feature_lv5_bot_left, feature_lv5_bot_right), 3)

            feature_lv5 = torch.cat((feature_lv5_top, feature_lv5_bot), 2)

            residual_lv5_top_left_top = decoder_lv5_a(feature_lv5_top_left_top)
            residual_lv5_top_left_bot = decoder_lv5_a(feature_lv5_top_left_bot)
            residual_lv5_top_right_top = decoder_lv5_a(feature_lv5_top_right_top)
            residual_lv5_top_right_bot = decoder_lv5_a(feature_lv5_top_right_bot)
            residual_lv5_bot_left_top = decoder_lv5_a(feature_lv5_bot_left_top)
            residual_lv5_bot_left_bot = decoder_lv5_a(feature_lv5_bot_left_bot)
            residual_lv5_bot_right_top = decoder_lv5_a(feature_lv5_bot_right_top)
            residual_lv5_bot_right_bot = decoder_lv5_a(feature_lv5_bot_right_bot)

            residual_lv5_top_left_left = decoder_lv5_b(feature_lv5_top_left_left)
            residual_lv5_top_left_right = decoder_lv5_b(feature_lv5_top_left_right)
            residual_lv5_top_right_left = decoder_lv5_b(feature_lv5_top_right_left)
            residual_lv5_top_right_right = decoder_lv5_b(feature_lv5_top_right_right)
            residual_lv5_bot_left_left = decoder_lv5_b(feature_lv5_bot_left_left)
            residual_lv5_bot_left_right = decoder_lv5_b(feature_lv5_bot_left_right)
            residual_lv5_bot_right_left = decoder_lv5_b(feature_lv5_bot_right_left)
            residual_lv5_bot_right_right = decoder_lv5_b(feature_lv5_bot_right_right)

            # residual_lv5_top_left = torch.cat((residual_lv5_top_left_top, residual_lv5_top_left_bot), 2)
            # residual_lv5_top_right = torch.cat((residual_lv5_top_right_top, residual_lv5_top_right_bot), 2)
            # residual_lv5_bot_right = torch.cat((residual_lv5_bot_right_top, residual_lv5_bot_right_bot), 2)
            # residual_lv5_bot_left = torch.cat((residual_lv5_bot_left_top, residual_lv5_bot_left_bot), 2)
            #
            # residual_lv5_top = torch.cat((residual_lv5_top_right, residual_lv5_top_left), 3)
            # residual_lv5_bot = torch.cat((residual_lv5_bot_right, residual_lv5_bot_left), 3)

            feature_lv4_1 = encoder_lv4_a(images_lv4_1 + residual_lv5_top_left_top)
            feature_lv4_2 = encoder_lv4_a(images_lv4_2 + residual_lv5_top_left_bot)
            feature_lv4_3 = encoder_lv4_a(images_lv4_3 + residual_lv5_top_right_top)
            feature_lv4_4 = encoder_lv4_a(images_lv4_4 + residual_lv5_top_right_bot)
            feature_lv4_5 = encoder_lv4_a(images_lv4_5 + residual_lv5_bot_left_top)
            feature_lv4_6 = encoder_lv4_a(images_lv4_6 + residual_lv5_bot_left_bot)
            feature_lv4_7 = encoder_lv4_a(images_lv4_7 + residual_lv5_bot_right_top)
            feature_lv4_8 = encoder_lv4_a(images_lv4_8 + residual_lv5_bot_right_bot)

            feature_lv4_9 = encoder_lv4_b(images_lv4_9 + residual_lv5_top_left_left)
            feature_lv4_10 = encoder_lv4_b(images_lv4_10 + residual_lv5_top_left_right)
            feature_lv4_11 = encoder_lv4_b(images_lv4_11 + residual_lv5_top_right_left)
            feature_lv4_12 = encoder_lv4_b(images_lv4_12 + residual_lv5_top_right_right)
            feature_lv4_13 = encoder_lv4_b(images_lv4_13 + residual_lv5_bot_left_left)
            feature_lv4_14 = encoder_lv4_b(images_lv4_14 + residual_lv5_bot_left_right)
            feature_lv4_15 = encoder_lv4_b(images_lv4_15 + residual_lv5_bot_right_left)
            feature_lv4_16 = encoder_lv4_b(images_lv4_16 + residual_lv5_bot_right_right)

            # -------------------
            # gtfeature_lv4_1 = encoder_lv4_a(gt_lv4_1)
            # gtfeature_lv4_2 = encoder_lv4_a(gt_lv4_2)
            # gtfeature_lv4_3 = encoder_lv4_a(gt_lv4_3)
            # gtfeature_lv4_4 = encoder_lv4_a(gt_lv4_4)
            # gtfeature_lv4_5 = encoder_lv4_a(gt_lv4_5)
            # gtfeature_lv4_6 = encoder_lv4_a(gt_lv4_6)
            # gtfeature_lv4_7 = encoder_lv4_a(gt_lv4_7)
            # gtfeature_lv4_8 = encoder_lv4_a(gt_lv4_8)
            # gtfeature_lv4_9 = encoder_lv4_b(gt_lv4_9)
            # gtfeature_lv4_10 = encoder_lv4_b(gt_lv4_10)
            # gtfeature_lv4_11 = encoder_lv4_b(gt_lv4_11)
            # gtfeature_lv4_12 = encoder_lv4_b(gt_lv4_12)
            # gtfeature_lv4_13 = encoder_lv4_b(gt_lv4_13)
            # gtfeature_lv4_14 = encoder_lv4_b(gt_lv4_14)
            # gtfeature_lv4_15 = encoder_lv4_b(gt_lv4_15)
            # gtfeature_lv4_16 = encoder_lv4_b(gt_lv4_16)

            # ----------------------------

            feature_lv4_top_left_a = torch.cat((feature_lv4_1, feature_lv4_2), 2) + feature_lv5_top_left
            feature_lv4_top_right_a = torch.cat((feature_lv4_3, feature_lv4_4), 2) + feature_lv5_top_right
            feature_lv4_bot_left_a = torch.cat((feature_lv4_5, feature_lv4_6), 2) + feature_lv5_bot_left
            feature_lv4_bot_right_a = torch.cat((feature_lv4_7, feature_lv4_8), 2) + feature_lv5_bot_right

            feature_lv4_top_left_b = torch.cat((feature_lv4_9, feature_lv4_10), 3) + feature_lv5_top_left
            feature_lv4_top_right_b = torch.cat((feature_lv4_11, feature_lv4_12), 3) + feature_lv5_top_right
            feature_lv4_bot_left_b = torch.cat((feature_lv4_13, feature_lv4_14), 3) + feature_lv5_bot_left
            feature_lv4_bot_right_b = torch.cat((feature_lv4_15, feature_lv4_16), 3) + feature_lv5_bot_right

            feature_lv4_top_a = torch.cat((feature_lv4_top_left_a, feature_lv4_top_right_a), 3)
            feature_lv4_bot_a = torch.cat((feature_lv4_bot_left_a, feature_lv4_bot_right_a), 3)
            feature_lv4_right_a = torch.cat((feature_lv4_top_right_a, feature_lv4_bot_right_a), 2)
            feature_lv4_left_a = torch.cat((feature_lv4_top_left_a, feature_lv4_bot_left_a), 2)

            feature_lv4_top_b = torch.cat((feature_lv4_top_left_b, feature_lv4_top_right_b), 3)
            feature_lv4_bot_b = torch.cat((feature_lv4_bot_left_b, feature_lv4_bot_right_b), 3)
            feature_lv4_right_b = torch.cat((feature_lv4_top_right_b, feature_lv4_bot_right_b), 2)
            feature_lv4_left_b = torch.cat((feature_lv4_top_left_b, feature_lv4_bot_left_b), 2)

            residual_lv4_top_left_a = decoder_lv4(feature_lv4_top_left_a)
            residual_lv4_top_right_a = decoder_lv4(feature_lv4_top_right_a)
            residual_lv4_bot_left_a = decoder_lv4(feature_lv4_bot_left_a)
            residual_lv4_bot_right_a = decoder_lv4(feature_lv4_bot_right_a)

            residual_lv4_top_left_b = decoder_lv4(feature_lv4_top_left_b)
            residual_lv4_top_right_b = decoder_lv4(feature_lv4_top_right_b)
            residual_lv4_bot_left_b = decoder_lv4(feature_lv4_bot_left_b)
            residual_lv4_bot_right_b = decoder_lv4(feature_lv4_bot_right_b)

            feature_lv3_1 = encoder_lv3(images_lv3_1 + residual_lv4_top_left_a + residual_lv4_top_left_b)
            feature_lv3_2 = encoder_lv3(images_lv3_2 + residual_lv4_top_right_a + residual_lv4_top_right_b)
            feature_lv3_3 = encoder_lv3(images_lv3_3 + residual_lv4_bot_left_a + residual_lv4_bot_left_b)
            feature_lv3_4 = encoder_lv3(images_lv3_4 + residual_lv4_bot_right_a + residual_lv4_bot_right_b)

            # -----------------------
            gtfeature_lv3_1 = encoder_lv3(gt_lv3_1 + residual_lv4_top_left_a + residual_lv4_top_left_b)
            gtfeature_lv3_2 = encoder_lv3(gt_lv3_2 + residual_lv4_top_right_a + residual_lv4_top_right_b)
            gtfeature_lv3_3 = encoder_lv3(gt_lv3_3 + residual_lv4_bot_left_a + residual_lv4_bot_left_b)
            gtfeature_lv3_4 = encoder_lv3(gt_lv3_4 + residual_lv4_bot_right_a + residual_lv4_bot_right_b)

            gtfeature_lv3_top = torch.cat((gtfeature_lv3_1, gtfeature_lv3_2), 3)
            gtfeature_lv3_bot = torch.cat((gtfeature_lv3_3, gtfeature_lv3_4), 3)

            gtfeature_lv3 = torch.cat((gtfeature_lv3_top, gtfeature_lv3_bot), 2)

            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top_a + feature_lv4_top_b
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot_a + feature_lv4_bot_b

            feature_lv3_right = torch.cat((feature_lv3_1, feature_lv3_3), 2) + feature_lv4_right_a + feature_lv4_right_b
            feature_lv3_left = torch.cat((feature_lv3_2, feature_lv3_4), 2) + feature_lv4_left_a + feature_lv4_left_b

            feature_lv3 = torch.cat((feature_lv3_right, feature_lv3_left), 3)

            residual_lv3_top = decoder_lv3_a(feature_lv3_top)
            residual_lv3_bot = decoder_lv3_a(feature_lv3_bot)

            residual_lv3_left = decoder_lv3_b(feature_lv3_left)
            residual_lv3_right = decoder_lv3_b(feature_lv3_right)

            feature_lv2_1 = encoder_lv2_a(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2_a(images_lv2_2 + residual_lv3_bot)

            feature_lv2_3 = encoder_lv2_b(images_lv2_3 + residual_lv3_left)
            feature_lv2_4 = encoder_lv2_b(images_lv2_4 + residual_lv3_right)

            # --------------------------

            # gtfeature_lv2_1 = encoder_lv2_a(gt_lv2_1)
            # gtfeature_lv2_2 = encoder_lv2_a(gt_lv2_2)
            #
            # gtfeature_lv2_3 = encoder_lv2_b(gt_lv2_3)
            # gtfeature_lv2_4 = encoder_lv2_b(gt_lv2_4)

            feature_lv2_a = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            feature_lv2_b = torch.cat((feature_lv2_3, feature_lv2_4), 3) + feature_lv3

            residual_lv2_a = decoder_lv2(feature_lv2_a)
            residual_lv2_b = decoder_lv2(feature_lv2_b)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2_a + residual_lv2_b)
            gtfeature_lv1 = encoder_lv1(gt_lv1)

            deblur_image = decoder_lv1(feature_lv1 + feature_lv2_a + feature_lv2_b)

            loss_fn = UnderwaterLoss1(reduction='mean')
            loss_fnn = UnderwaterLoss2(reduction='mean')
            common_loss = loss_fnn(deblur_image, gt)
            # common_loss = mse(deblur_image, gt)
            perceptual_loss = loss_fn(feature_lv1, gtfeature_lv1) + loss_fn(feature_lv3, gtfeature_lv3) + loss_fn(
                feature_lv5, gtfeature_lv5)
            loss = common_loss + 0.005 * perceptual_loss + mse(deblur_image, gt)
            # sum_loss = sum_loss + loss

            encoder_lv1.zero_grad()
            encoder_lv2_a.zero_grad()
            encoder_lv2_b.zero_grad()
            encoder_lv3.zero_grad()
            encoder_lv4_a.zero_grad()
            encoder_lv4_b.zero_grad()
            encoder_lv5.zero_grad()
            # -------
            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3_a.zero_grad()
            decoder_lv3_b.zero_grad()
            decoder_lv4.zero_grad()
            decoder_lv5_a.zero_grad()
            decoder_lv5_b.zero_grad()

            loss.backward()
            # ------------------------------------------------------------------
            encoder_lv1_optim.step()
            encoder_lv2_a_optim.step()
            encoder_lv2_b_optim.step()
            encoder_lv3_optim.step()
            encoder_lv4_a_optim.step()
            encoder_lv4_b_optim.step()
            encoder_lv5_optim.step()
            # ------
            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_a_optim.step()
            decoder_lv3_b_optim.step()
            decoder_lv4_optim.step()
            decoder_lv5_a_optim.step()
            decoder_lv5_b_optim.step()

            if (iteration + 1) % 10 == 0:
                stop = time.time()
                print("epoch:", epoch, "iteration:", iteration + 1, "loss:%.4f" % loss.item(),
                      'time:%.4f' % (stop - start))
                start = time.time()

            if (iteration + 1) % 900 == 0:
                print("loss:%.4f" % loss.item())

        if (epoch + 1) % 65535 == 0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
                os.system('mkdir ./checkpoints/' + METHOD + '/epoch' + str(epoch))

            print("Testing...")
            test_dataset = GoProDataset(
                blur_image_files='./datas/UMADD/test_blur_file.txt',
                sharp_image_files='./datas/UMADD/test_sharp_file.txt',
                root_dir='./datas/UMADD',
                transform=transforms.Compose([
                    transforms.ToTensor()
                ]))
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_time = 0
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():
                    start = time.time()

                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)

                    H = images_lv1.size(2)
                    W = images_lv1.size(3)

                    images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
                    images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]

                    images_lv2_3 = images_lv1[:, :, :, 0:int(W / 2)]  # add
                    images_lv2_4 = images_lv1[:, :, :, int(W / 2):W]

                    images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
                    images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
                    images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
                    images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]

                    images_lv4_1 = images_lv3_1[:, :, 0:int(H / 4), :]
                    images_lv4_2 = images_lv3_1[:, :, int(H / 4):int(H / 2), :]
                    images_lv4_3 = images_lv3_2[:, :, 0:int(H / 4), :]
                    images_lv4_4 = images_lv3_2[:, :, int(H / 4):int(H / 2), :]
                    images_lv4_5 = images_lv3_3[:, :, 0:int(H / 4), :]
                    images_lv4_6 = images_lv3_3[:, :, int(H / 4):int(H / 2), :]
                    images_lv4_7 = images_lv3_4[:, :, 0:int(H / 4), :]
                    images_lv4_8 = images_lv3_4[:, :, int(H / 4):int(H / 2), :]

                    images_lv4_9 = images_lv3_1[:, :, :, 0:int(W / 4)]
                    images_lv4_10 = images_lv3_1[:, :, :, int(W / 4):int(W / 2)]
                    images_lv4_11 = images_lv3_2[:, :, :, 0:int(W / 4)]
                    images_lv4_12 = images_lv3_2[:, :, :, int(W / 4):int(W / 2)]
                    images_lv4_13 = images_lv3_3[:, :, :, 0:int(W / 4)]
                    images_lv4_14 = images_lv3_3[:, :, :, int(W / 4):int(W / 2)]
                    images_lv4_15 = images_lv3_4[:, :, :, 0:int(W / 4)]
                    images_lv4_16 = images_lv3_4[:, :, :, int(W / 4):int(W / 2)]

                    images_lv5_1 = images_lv4_1[:, :, :, 0:int(W / 4)]
                    images_lv5_2 = images_lv4_1[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_3 = images_lv4_2[:, :, :, 0:int(W / 4)]
                    images_lv5_4 = images_lv4_2[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_5 = images_lv4_3[:, :, :, 0:int(W / 4)]
                    images_lv5_6 = images_lv4_3[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_7 = images_lv4_4[:, :, :, 0:int(W / 4)]
                    images_lv5_8 = images_lv4_4[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_9 = images_lv4_5[:, :, :, 0:int(W / 4)]
                    images_lv5_10 = images_lv4_5[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_11 = images_lv4_6[:, :, :, 0:int(W / 4)]
                    images_lv5_12 = images_lv4_6[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_13 = images_lv4_7[:, :, :, 0:int(W / 4)]
                    images_lv5_14 = images_lv4_7[:, :, :, int(W / 4):int(W / 2)]
                    images_lv5_15 = images_lv4_8[:, :, :, 0:int(W / 4)]
                    images_lv5_16 = images_lv4_8[:, :, :, int(W / 4):int(W / 2)]

                    feature_lv5_1 = encoder_lv5(images_lv5_1)
                    feature_lv5_2 = encoder_lv5(images_lv5_2)
                    feature_lv5_3 = encoder_lv5(images_lv5_3)
                    feature_lv5_4 = encoder_lv5(images_lv5_4)
                    feature_lv5_5 = encoder_lv5(images_lv5_5)
                    feature_lv5_6 = encoder_lv5(images_lv5_6)
                    feature_lv5_7 = encoder_lv5(images_lv5_7)
                    feature_lv5_8 = encoder_lv5(images_lv5_8)
                    feature_lv5_9 = encoder_lv5(images_lv5_9)
                    feature_lv5_10 = encoder_lv5(images_lv5_10)
                    feature_lv5_11 = encoder_lv5(images_lv5_11)
                    feature_lv5_12 = encoder_lv5(images_lv5_12)
                    feature_lv5_13 = encoder_lv5(images_lv5_13)
                    feature_lv5_14 = encoder_lv5(images_lv5_14)
                    feature_lv5_15 = encoder_lv5(images_lv5_15)
                    feature_lv5_16 = encoder_lv5(images_lv5_16)

                    feature_lv5_top_left_top = torch.cat((feature_lv5_1, feature_lv5_2), 3)
                    feature_lv5_top_left_bot = torch.cat((feature_lv5_3, feature_lv5_4), 3)
                    feature_lv5_top_right_top = torch.cat((feature_lv5_5, feature_lv5_6), 3)
                    feature_lv5_top_right_bot = torch.cat((feature_lv5_7, feature_lv5_8), 3)
                    feature_lv5_bot_left_top = torch.cat((feature_lv5_9, feature_lv5_10), 3)
                    feature_lv5_bot_left_bot = torch.cat((feature_lv5_11, feature_lv5_12), 3)
                    feature_lv5_bot_right_top = torch.cat((feature_lv5_13, feature_lv5_14), 3)
                    feature_lv5_bot_right_bot = torch.cat((feature_lv5_15, feature_lv5_16), 3)

                    feature_lv5_top_left_left = torch.cat((feature_lv5_1, feature_lv5_3), 2)
                    feature_lv5_top_left_right = torch.cat((feature_lv5_2, feature_lv5_4), 2)
                    feature_lv5_top_right_left = torch.cat((feature_lv5_5, feature_lv5_7), 2)
                    feature_lv5_top_right_right = torch.cat((feature_lv5_6, feature_lv5_8), 2)
                    feature_lv5_bot_left_left = torch.cat((feature_lv5_9, feature_lv5_11), 2)
                    feature_lv5_bot_left_right = torch.cat((feature_lv5_10, feature_lv5_12), 2)
                    feature_lv5_bot_right_left = torch.cat((feature_lv5_13, feature_lv5_15), 2)
                    feature_lv5_bot_right_right = torch.cat((feature_lv5_14, feature_lv5_16), 2)

                    feature_lv5_top_left = torch.cat((feature_lv5_top_left_top, feature_lv5_top_left_bot), 2)
                    feature_lv5_top_right = torch.cat((feature_lv5_top_right_top, feature_lv5_top_right_bot), 2)
                    feature_lv5_bot_left = torch.cat((feature_lv5_bot_left_top, feature_lv5_bot_left_bot), 2)
                    feature_lv5_bot_right = torch.cat((feature_lv5_bot_right_top, feature_lv5_bot_right_bot), 2)

                    residual_lv5_top_left_top = decoder_lv5_a(feature_lv5_top_left_top)
                    residual_lv5_top_left_bot = decoder_lv5_a(feature_lv5_top_left_bot)
                    residual_lv5_top_right_top = decoder_lv5_a(feature_lv5_top_right_top)
                    residual_lv5_top_right_bot = decoder_lv5_a(feature_lv5_top_right_bot)
                    residual_lv5_bot_left_top = decoder_lv5_a(feature_lv5_bot_left_top)
                    residual_lv5_bot_left_bot = decoder_lv5_a(feature_lv5_bot_left_bot)
                    residual_lv5_bot_right_top = decoder_lv5_a(feature_lv5_bot_right_top)
                    residual_lv5_bot_right_bot = decoder_lv5_a(feature_lv5_bot_right_bot)

                    residual_lv5_top_left_left = decoder_lv5_b(feature_lv5_top_left_left)
                    residual_lv5_top_left_right = decoder_lv5_b(feature_lv5_top_left_right)
                    residual_lv5_top_right_left = decoder_lv5_b(feature_lv5_top_right_left)
                    residual_lv5_top_right_right = decoder_lv5_b(feature_lv5_top_right_right)
                    residual_lv5_bot_left_left = decoder_lv5_b(feature_lv5_bot_left_left)
                    residual_lv5_bot_left_right = decoder_lv5_b(feature_lv5_bot_left_right)
                    residual_lv5_bot_right_left = decoder_lv5_b(feature_lv5_bot_right_left)
                    residual_lv5_bot_right_right = decoder_lv5_b(feature_lv5_bot_right_right)

                    feature_lv4_1 = encoder_lv4_a(images_lv4_1 + residual_lv5_top_left_top)
                    feature_lv4_2 = encoder_lv4_a(images_lv4_2 + residual_lv5_top_left_bot)
                    feature_lv4_3 = encoder_lv4_a(images_lv4_3 + residual_lv5_top_right_top)
                    feature_lv4_4 = encoder_lv4_a(images_lv4_4 + residual_lv5_top_right_bot)
                    feature_lv4_5 = encoder_lv4_a(images_lv4_5 + residual_lv5_bot_left_top)
                    feature_lv4_6 = encoder_lv4_a(images_lv4_6 + residual_lv5_bot_left_bot)
                    feature_lv4_7 = encoder_lv4_a(images_lv4_7 + residual_lv5_bot_right_top)
                    feature_lv4_8 = encoder_lv4_a(images_lv4_8 + residual_lv5_bot_right_bot)

                    feature_lv4_9 = encoder_lv4_b(images_lv4_9 + residual_lv5_top_left_left)
                    feature_lv4_10 = encoder_lv4_b(images_lv4_10 + residual_lv5_top_left_right)
                    feature_lv4_11 = encoder_lv4_b(images_lv4_11 + residual_lv5_top_right_left)
                    feature_lv4_12 = encoder_lv4_b(images_lv4_12 + residual_lv5_top_right_right)
                    feature_lv4_13 = encoder_lv4_b(images_lv4_13 + residual_lv5_bot_left_left)
                    feature_lv4_14 = encoder_lv4_b(images_lv4_14 + residual_lv5_bot_left_right)
                    feature_lv4_15 = encoder_lv4_b(images_lv4_15 + residual_lv5_bot_right_left)
                    feature_lv4_16 = encoder_lv4_b(images_lv4_16 + residual_lv5_bot_right_right)

                    feature_lv4_top_left_a = torch.cat((feature_lv4_1, feature_lv4_2), 2) + feature_lv5_top_left
                    feature_lv4_top_right_a = torch.cat((feature_lv4_3, feature_lv4_4), 2) + feature_lv5_top_right
                    feature_lv4_bot_left_a = torch.cat((feature_lv4_5, feature_lv4_6), 2) + feature_lv5_bot_left
                    feature_lv4_bot_right_a = torch.cat((feature_lv4_7, feature_lv4_8), 2) + feature_lv5_bot_right

                    feature_lv4_top_left_b = torch.cat((feature_lv4_9, feature_lv4_10), 3) + feature_lv5_top_left
                    feature_lv4_top_right_b = torch.cat((feature_lv4_11, feature_lv4_12), 3) + feature_lv5_top_right
                    feature_lv4_bot_left_b = torch.cat((feature_lv4_13, feature_lv4_14), 3) + feature_lv5_bot_left
                    feature_lv4_bot_right_b = torch.cat((feature_lv4_15, feature_lv4_16), 3) + feature_lv5_bot_right

                    feature_lv4_top_a = torch.cat((feature_lv4_top_left_a, feature_lv4_top_right_a), 3)
                    feature_lv4_bot_a = torch.cat((feature_lv4_bot_left_a, feature_lv4_bot_right_a), 3)
                    feature_lv4_right_a = torch.cat((feature_lv4_top_right_a, feature_lv4_bot_right_a), 2)
                    feature_lv4_left_a = torch.cat((feature_lv4_top_left_a, feature_lv4_bot_left_a), 2)

                    feature_lv4_top_b = torch.cat((feature_lv4_top_left_b, feature_lv4_top_right_b), 3)
                    feature_lv4_bot_b = torch.cat((feature_lv4_bot_left_b, feature_lv4_bot_right_b), 3)
                    feature_lv4_right_b = torch.cat((feature_lv4_top_right_b, feature_lv4_bot_right_b), 2)
                    feature_lv4_left_b = torch.cat((feature_lv4_top_left_b, feature_lv4_bot_left_b), 2)

                    residual_lv4_top_left_a = decoder_lv4(feature_lv4_top_left_a)
                    residual_lv4_top_right_a = decoder_lv4(feature_lv4_top_right_a)
                    residual_lv4_bot_left_a = decoder_lv4(feature_lv4_bot_left_a)
                    residual_lv4_bot_right_a = decoder_lv4(feature_lv4_bot_right_a)

                    residual_lv4_top_left_b = decoder_lv4(feature_lv4_top_left_b)
                    residual_lv4_top_right_b = decoder_lv4(feature_lv4_top_right_b)
                    residual_lv4_bot_left_b = decoder_lv4(feature_lv4_bot_left_b)
                    residual_lv4_bot_right_b = decoder_lv4(feature_lv4_bot_right_b)

                    feature_lv3_1 = encoder_lv3(images_lv3_1 + residual_lv4_top_left_a + residual_lv4_top_left_b)
                    feature_lv3_2 = encoder_lv3(images_lv3_2 + residual_lv4_top_right_a + residual_lv4_top_right_b)
                    feature_lv3_3 = encoder_lv3(images_lv3_3 + residual_lv4_bot_left_a + residual_lv4_bot_left_b)
                    feature_lv3_4 = encoder_lv3(images_lv3_4 + residual_lv4_bot_right_a + residual_lv4_bot_right_b)

                    feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2),
                                                3) + feature_lv4_top_a + feature_lv4_top_b
                    feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4),
                                                3) + feature_lv4_bot_a + feature_lv4_bot_b

                    feature_lv3_right = torch.cat((feature_lv3_1, feature_lv3_3),
                                                  2) + feature_lv4_right_a + feature_lv4_right_b
                    feature_lv3_left = torch.cat((feature_lv3_2, feature_lv3_4),
                                                 2) + feature_lv4_left_a + feature_lv4_left_b

                    feature_lv3 = torch.cat((feature_lv3_right, feature_lv3_left), 3)

                    residual_lv3_top = decoder_lv3_a(feature_lv3_top)
                    residual_lv3_bot = decoder_lv3_a(feature_lv3_bot)

                    residual_lv3_left = decoder_lv3_b(feature_lv3_left)
                    residual_lv3_right = decoder_lv3_b(feature_lv3_right)

                    feature_lv2_1 = encoder_lv2_a(images_lv2_1 + residual_lv3_top)
                    feature_lv2_2 = encoder_lv2_a(images_lv2_2 + residual_lv3_bot)

                    feature_lv2_3 = encoder_lv2_b(images_lv2_3 + residual_lv3_left)
                    feature_lv2_4 = encoder_lv2_b(images_lv2_4 + residual_lv3_right)

                    feature_lv2_a = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
                    feature_lv2_b = torch.cat((feature_lv2_3, feature_lv2_4), 3) + feature_lv3

                    residual_lv2_a = decoder_lv2(feature_lv2_a)
                    residual_lv2_b = decoder_lv2(feature_lv2_b)

                    feature_lv1 = encoder_lv1(
                        images_lv1 + residual_lv2_a + residual_lv2_b) + feature_lv2_a + feature_lv2_b
                    deblur_image = decoder_lv1(feature_lv1)

                    stop = time.time()
                    test_time += stop - start
                    print('RunTime:%.4f' % (stop - start), '  Average Runtime:%.4f' % (test_time / (iteration + 1)))
                    save_deblur_images(deblur_image.data + 0.5, iteration, epoch)

        torch.save(encoder_lv1.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv1.pkl"))
        torch.save(encoder_lv2_a.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv2_a.pkl"))
        torch.save(encoder_lv2_b.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv2_b.pkl"))
        torch.save(encoder_lv3.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv3.pkl"))
        torch.save(encoder_lv4_a.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv4_a.pkl"))
        torch.save(encoder_lv4_b.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv4_b.pkl"))
        torch.save(encoder_lv5.state_dict(), str('./checkpoints/' + METHOD + "/encoder_lv5.pkl"))
        # ---------------------
        torch.save(decoder_lv1.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv1.pkl"))
        torch.save(decoder_lv2.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv2.pkl"))
        torch.save(decoder_lv3_a.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv3_a.pkl"))
        torch.save(decoder_lv3_b.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv3_b.pkl"))
        torch.save(decoder_lv4.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv4.pkl"))
        torch.save(decoder_lv5_a.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv5_a.pkl"))
        torch.save(decoder_lv5_b.state_dict(), str('./checkpoints/' + METHOD + "/decoder_lv5_b.pkl"))


if __name__ == '__main__':
    main()
