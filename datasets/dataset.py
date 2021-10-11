import torch
from torch import tensor
from torch.utils.data import Dataset
from utils.functions import imshow

import time
import numpy as np
from PIL import Image

class DatasetFM(Dataset):
  tensor_view = (1, 28, 28) #For FashionMnist
  # tensor_view = (3, 32, 32)   #For Cifar10

  def __init__(self, dataset, transforms=None):
		# dataset = read_csv(path, sep=',', header=None).values
    self.transforms = transforms
    self.data = []
    self.labels = dataset[:, -1]
    self.label_set = set(self.labels)

    # Convert textual labels to int labels
    # labels = dataset[:, -1]
    # label_set = set(labels)
    # text2int = {text_label: idx for idx, text_label in enumerate(label_set)}
    # print(text2int)
    # time.sleep(5)
    # for idx, item in enumerate(labels):
    #   labels[idx] = text2int[item]
  
    for idx, s in enumerate(dataset):

      # show image
      # img = (tensor(s[:-1], dtype=torch.float) / 255.).view(self.tensor_view)
      # imshow(img)

      x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
      y = tensor(self.labels[idx], dtype=torch.long)
      self.data.append((x, y))

  def __getitem__(self, index):
    if self.transforms != None:
      sample, label = self.data[index]
      sample = self.transforms(sample)
      return sample, label
    else:
      return self.data[index]

  def __len__(self):
    return len(self.data)
  
  def get_sample_per_class(self, n_samples=1):
    
    ## == split data by classes =================
    class_data = {}
    for class_label in set(self.labels):
      class_data[class_label] = []
    
    for idx, (sample, label) in enumerate(self.data):
      class_data[label.item()].append(sample)
    
    imgs_list = []
    labels_list = []
    for label, data in class_data.items():
      idxs = np.random.choice(len(data), n_samples, replace=False)
      imgs = [data[idx] for idx in idxs]
      imgs_list.extend(imgs)
      labels_list.extend([label for i in range(n_samples)])

    # x = torch.unsqueeze(torch.stack(imgs_list), 1)
    x = torch.stack(imgs_list)
    y = torch.tensor(labels_list)
    return x, y