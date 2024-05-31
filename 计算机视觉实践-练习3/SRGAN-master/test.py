import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='/netG_epoch_2_80.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {'Set5': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    # 获取 sr_image 的尺寸
    target_size = sr_image.size()[2:]

    # 将 hr_image 调整为与 sr_image 相同的大小
    hr_image = F.interpolate(hr_image, size=target_size, mode='bilinear', align_corners=False)

    mse = ((hr_image - sr_image) ** 2).mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                     image_name.split('.')[-1], padding=5)

    # save psnr\ssim
    results[image_name.split('_')[0]]['psnr'].append(psnr)
    results[image_name.split('_')[0]]['ssim'].append(ssim)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')