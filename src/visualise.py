import os

import numpy as np
from PIL import Image

import torch	

from .models import SRCNN
from .utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


def visualise(config):
	device = torch.device("cuda" if config['cuda'] else "gpu")
	model = SRCNN().to(device)
	model.load_state_dict(torch.load(config['model_path']))

	model.eval()
	torch.no_grad()

	image = Image.open(config['image_file']).convert('RGB')
	image_width = (image.width // config['scale']) * config['scale']	
	image_height = (image.height // config['scale']) * config['scale']	
	image = image.resize((image_width, image_height), resample=Image.BICUBIC)
	image = image.resize((image.width // config['scale']	, image.height // config['scale']	), resample=Image.BICUBIC)
	image = image.resize((image.width * config['scale']	, image.height * config['scale']	), resample=Image.BICUBIC)
	image.save(os.path.join(config['results'],f"BICUBIC_{config['image_file']}"))
	image = np.array(image).astype(np.float32)
	image = convert_rgb_to_ycbcr(image)
	
	y = image[..., 0]/255
	y = torch.from_numpy(y).to(device)
	y = y.unsqueeze(0).unsqueeze(0)

	pred = model(y).clamp(0.0,1.0)
	psnr = calc_psnr(y,pred)
	print(f"psnr:-{psnr}")

	pred = pred.mul(255.0).cpu().detach().numpy().squeeze(0).squeeze(0)

	output = np.array([pred,image[..., 1],image[..., 2]]).transpose([1, 2, 0])
	output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
	output = Image	.fromarray(output)
	output.save(os.path.join(config['results'],f'SRCNN_1_{config["image_file"]}'))

