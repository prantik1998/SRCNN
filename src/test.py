import os

import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data

from .models import SRCNN
# from .utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from .dataset import Set5Dataset
from .utils import AverageMeter,calc_psnr

def test(config):
	device = torch.device("cuda" if config["cuda"] else "gpu")
	model = SRCNN().to(device)
	model.load_state_dict(torch.load(config['model_path']))
	model.eval()
	torch.no_grad()
	epoch_psnr = AverageMeter()
	dataset = Set5Dataset(config['evaldata'])
	dataloader = data.DataLoader(dataset,1)
	for (image,labels) in dataloader:
		image,labels = image.to(device),labels.to(device)
		output = model(image).clamp(0.0,1.0)
		epoch_psnr.update(calc_psnr(output,labels),len(image))
	print(f"epoch_psnr:-{epoch_psnr.avg}")


