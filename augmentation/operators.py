import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Resize
from augmentation.visualize import imshow
import torchvision.transforms.functional as TF
import random
from PIL import Image

class AddGaussianNoise(object):
  def __init__(self, mean=0., std=1.):
    self.std = std
    self.mean = mean
      
  def __call__(self, tensor):
    return tensor + torch.randn(tensor.size()) * self.std + self.mean
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def traditional_aug(support_images, query_images):
  
  transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(30),
    transforms.RandomCrop((26,26)),
    transforms.Resize((28, 28)),
    # transforms.RandomVerticalFlip(0.4), 
    # transforms.RandomHorizontalFlip(0.4),
    # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]) 
    # AddGaussianNoise(0.1, 0.08),
    # transforms.RandomErasing(),
  ])
  
  support_images = support_images.permute(0, 2, 3, 1).detach().cpu().numpy().copy()
  aug_support_images = []
  for image in support_images:    
    # org_image = transforms.ToTensor()(image).unsqueeze_(0)
    image = transform(np.array(image)).unsqueeze_(0)
    aug_support_images.append(image)
    # imshow(torch.cat([org_image, image]))
  aug_support_images = torch.cat(aug_support_images)
  
  query_images = query_images.permute(0, 2, 3, 1).detach().cpu().numpy()
  aug_query_images = []
  for image in query_images:    
    # org_image = transforms.ToTensor()(image).unsqueeze_(0)
    image = transform(np.array(image)).unsqueeze_(0)
    aug_query_images.append(image)
    # imshow(torch.cat([org_image, image]))
  aug_query_images = torch.cat(aug_query_images)

  return aug_support_images, aug_query_images

# -----------------------------------------------------------------------------
# Limeng Qiao, Yemin Shi, Jia Li, Yaowei Wang, Tiejun Huang and Yonghong Tian.
# Transductive Episodic-Wise Adaptive Metric for Few-Shot Learning, ICCV, 2019.
# -----------------------------------------------------------------------------
def TIM_A(x, alpha, device):
  #lam should be: [0.5, 1]
  if alpha > 0:
    lam = np.random.beta(alpha, alpha)
    lam = np.maximum(lam, 1 - lam)
  else:
    lam = 1

  batch_size = x.shape[0]
  index = torch.randperm(batch_size).to(device)
  mixed_x = lam * x + (1 - lam) * x[index]
  return mixed_x


def TIM_S(x, alpha, device):
  #lam should be: [0.5, 1.5]
  if alpha > 0:
    lam = np.random.beta(alpha, alpha)
    lam = np.maximum(lam, 1 - lam)
    lam += 0.5
  else:
    lam = 1
  
  batch_size = x.shape[0]
  index = torch.randperm(batch_size).to(device)
  mixed_x = lam * x + (1 - lam) * x[index]
  return mixed_x

