import torch
import torch.nn as nn
import math

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
			m.bias.data.zero_()
	elif classname.find('BatchNorm') != -1:
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif classname.find('Linear') != -1:
		n = m.weight.size(1)
		m.weight.data.normal_(0, 0.01)
		m.bias.data = torch.ones(m.bias.data.size())

class CNNEncoder(nn.Module):
	"""docstring for ClassName"""
	def __init__(self, args):
		super(CNNEncoder, self).__init__()
		self.layer1 = nn.Sequential(
										nn.Conv2d(1,64,kernel_size=3,padding=0),
										nn.BatchNorm2d(64, momentum=1, affine=True),
										nn.ReLU(),
										nn.MaxPool2d(2),
										nn.Dropout(args.dropout))
		self.layer2 = nn.Sequential(
										nn.Conv2d(64,64,kernel_size=3,padding=0),
										nn.BatchNorm2d(64, momentum=1, affine=True),
										nn.ReLU(),
										nn.MaxPool2d(2),
										nn.Dropout(args.dropout))
		self.layer3 = nn.Sequential(
										nn.Conv2d(64,64,kernel_size=3,padding=1),
										nn.BatchNorm2d(64, momentum=1, affine=True),
										nn.ReLU(),
										nn.Dropout(args.dropout))
		self.layer4 = nn.Sequential(
										nn.Conv2d(64,64,kernel_size=3,padding=1),
										nn.BatchNorm2d(64, momentum=1, affine=True),
										nn.ReLU(),
										nn.Dropout(args.dropout))

	def forward(self,x): #[bs, 1, 28, 28]
		out = self.layer1(x)   #[bs, 64, 13, 13]
		out = self.layer2(out) #[bs, 64, 5, 5]
		out = self.layer3(out) #[bs, 64, 5, 5]
		out = self.layer4(out) #[bs, 64, 5, 5]
		#out = out.view(out.size(0),-1)
		return out # 64

	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		self.layer1 = self.layer1.to(*args, **kwargs)
		self.layer2 = self.layer2.to(*args, **kwargs)
		self.layer3 = self.layer3.to(*args, **kwargs)
		self.layer4 = self.layer4.to(*args, **kwargs)

		return self
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)

