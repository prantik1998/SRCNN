import torch 
import torch.nn as nn

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN,self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2),nn.BatchNorm2d(64),nn.ReLU())
		self.layer2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),nn.BatchNorm2d(32),nn.ReLU())
		self.out =  nn.Conv2d(32,1, kernel_size=5, padding=5 // 2)

	def forward(self,x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.out(x)

		return x

if __name__=="__main__":
	model = SRCNN()
	x = torch.rand(1,3,64,64)
	print(x.size())