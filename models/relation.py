import torch
import torch.nn as nn
import time


class RelationNetwork(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self,input_size, hidden_size):
    super(RelationNetwork, self).__init__()
    self.layer1 = nn.Sequential(
                    nn.Conv2d(128,64,kernel_size=3,padding=1),
                    nn.BatchNorm2d(64, momentum=1, affine=True),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
    self.layer2 = nn.Sequential(
                    nn.Conv2d(64,64,kernel_size=3,padding=1),
                    nn.BatchNorm2d(64, momentum=1, affine=True),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
    self.fc1 = nn.Linear(input_size*1*1,hidden_size)
    self.fc2 = nn.Linear(hidden_size,1)

  def forward(self,x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0),-1)
    out = torch.relu(self.fc1(out))
    out = torch.sigmoid(self.fc2(out))
    return out

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    self.layer1 = self.layer1.to(*args, **kwargs)
    self.layer2 = self.layer2.to(*args, **kwargs)
    self.fc1 = self.fc1.to(*args, **kwargs)
    self.fc2 = self.fc2.to(*args, **kwargs)

    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)



class RelationNetworkFC(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self,args):
    super(RelationNetworkFC, self).__init__()
    
    self.fc1 = nn.Linear(2*args.hidden_dims,64)
    self.fc2 = nn.Linear(64,1)

  def forward(self,x):
    out = torch.relu(self.fc1(x))
    out = torch.sigmoid(self.fc2(out))
    return out

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    self.fc1 = self.fc1.to(*args, **kwargs)
    self.fc2 = self.fc2.to(*args, **kwargs)

    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)













class RelationMLP(nn.Module):
  
  def __init__(self, feature_size):
    super(RelationMLP, self).__init__()
    self.fc1 = nn.Linear(feature_size, 64)
    self.fc2 = nn.Linear(64, 1)
  
  def forward(self, x):
    out = torch.relu(self.fc1(x))
    out = torch.sigmoid(self.fc2(out))
    return out

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.fc1 = self.fc1.to(*args, **kwargs)
    self.fc2 = self.fc2.to(*args, **kwargs)
    return self
  
  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)


