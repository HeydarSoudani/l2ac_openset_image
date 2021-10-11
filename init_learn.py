import time
from pandas import read_csv

from datasets.dataset import DatasetFM
from trainers.train import train


def init_learn(model, mclassifer, args, device):

  ## == load train data from file ===
  train_data = read_csv(args.train_path, sep=',', header=None).values

  ### == Train baselines models ======
  train(model, mclassifer, train_data, args, device)


if __name__ == '__main__':
  pass