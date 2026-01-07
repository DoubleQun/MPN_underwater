import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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
from PIL import Image
from collections import OrderedDict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=2600)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=2)
parser.add_argument("-s", "--imagesize", type=int, default=256)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "124816MPN"
SAMPLE_DIR = "testPhoto"
EXPDIR = "MPNL1_test_res"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize


def save_images(images, name):
    dir_path = './test_results/' + EXPDIR
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = dir_path + "/" + name
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5 * math.sqrt(2. / n))
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

    encoder_lv1 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv2_a = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv2_b = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv3 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv4_a = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv4_b = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv5 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv1 = nn.DataParallel(encoder_lv1)
    encoder_lv2_a = nn.DataParallel(encoder_lv2_a)
    encoder_lv2_b = nn.DataParallel(encoder_lv2_b)
    encoder_lv3 = nn.DataParallel(encoder_lv3)
    encoder_lv4_a = nn.DataParallel(encoder_lv4_a)
    encoder_lv4_b = nn.DataParallel(encoder_lv4_b)
    encoder_lv5 = nn.DataParallel(encoder_lv5)

    decoder_lv1 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv2 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv3_a = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv3_b = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv4 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv5_a = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv5_b = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv1 = nn.DataParallel(decoder_lv1)
    decoder_lv2 = nn.DataParallel(decoder_lv2)
    decoder_lv3_a = nn.DataParallel(decoder_lv3_a)
    decoder_lv3_b = nn.DataParallel(decoder_lv3_b)
    decoder_lv4 = nn.DataParallel(decoder_lv4)
    decoder_lv5_a = nn.DataParallel(decoder_lv5_a)
    decoder_lv5_b = nn.DataParallel(decoder_lv5_b)

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

    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)

    iteration = 0.0
    test_time = 0.0
    for images_name in os.listdir(SAMPLE_DIR):
        with torch.no_grad():
            images_lv1 = transforms.ToTensor()(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB'))
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)
            start = time.time()
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

            feature_lv2_a = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            feature_lv2_b = torch.cat((feature_lv2_3, feature_lv2_4), 3) + feature_lv3

            residual_lv2_a = decoder_lv2(feature_lv2_a)
            residual_lv2_b = decoder_lv2(feature_lv2_b)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2_a + residual_lv2_b)
            deblur_image = decoder_lv1(feature_lv1 + feature_lv2_a + feature_lv2_b)

            # print(feature_lv1.shape)
            # print(feature_lv3.shape)

            stop = time.time()
            test_time += stop - start
            print('RunTime:%.4f' % (stop - start), '  Average Runtime:%.4f' % (test_time / (iteration + 1)))
            save_images(deblur_image.data + 0.5, images_name)
            iteration += 1


if __name__ == '__main__':
    main()
