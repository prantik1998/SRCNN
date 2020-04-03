import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from .dataset import Image91
from .models import SRCNN,weights_init
from .utils import AverageMeter

def train(config):
	dataset = Image91(config["traindata"])
	dataloader = data.DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=4)
	device = torch.device("cuda" if config['cuda'] else 'gpu')

	model = SRCNN().to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam([{'params': model.layer1.parameters()},{'params': model.layer2.parameters()},{'params': model.out.parameters(), 'lr':config['lr'] * 0.1}], lr=config['lr'])

	if 'model_path' in config.keys():
		print('Loading pretrained Model')
		model.load_state_dict(torch.load(config['model_path']))
	else:
		print('Initialising Model')
		model.apply(weights_init)

	for epoch in range(config['epochs']):
		model.train()
		epoch_losses = AverageMeter()
		for i,(image,labels) in enumerate(dataloader):
			image,labels = image.to(device),labels.to(device)
			output = model(image)
			loss = criterion(labels,output)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_losses.update(loss.item(), len(image))
			print(f"Epoch:{epoch+1},[{i}/{len(dataloader)}],Loss:-{loss.item()}")

		print(f"End of the Epoch")
		epoch_losses.show()
		if(epoch+1)%10 == 0:
			torch.save(model.state_dict(), os.path.join(config['pretrained'], 'epoch_{}.pth'.format(epoch)))
