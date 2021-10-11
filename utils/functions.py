from pandas import read_csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import time

def imshow(imgs):
  img = torchvision.utils.make_grid(imgs)
  # img = img / 2 + 0.5     # unnormalize
  npimg = img.detach().cpu().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def mean_std_calculator(nb_samples, dataloader):
  mean = 0.
  std = 0.
  for data,_ in dataloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
  mean /= nb_samples
  std /= nb_samples

  return mean, std


def set_novel_label(args):
  stream_data = read_csv(args.train_path, sep=',', header=None).values
  train_labels = stream_data[:, -1]
  seen_label = set(train_labels)
  stream_data = read_csv(args.test_path, sep=',', header=None).values

  for idx, data in enumerate(stream_data):
    label = data[-1]
    if label not in seen_label:
      stream_data[idx, -1] = 1

  pd.DataFrame(stream_data).to_csv('./data/cifar10_stream_novel.csv', header=None, index=None)


def mapping_text2int(data):
  samples = data[:, :-1].astype(int)
  labels = data[:, -1]
  label_set = set(labels)
  text2int = {text_label: idx for idx, text_label in enumerate(label_set)}
  
  for idx, item in enumerate(labels):
    labels[idx] = text2int[item]
  
  new_data = np.concatenate((samples, labels.astype(int)), axis=1)
  return new_data, text2int


if __name__ == '__main__':
  # set_novel_label()
  pass