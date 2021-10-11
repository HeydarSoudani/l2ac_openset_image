import os
import pickle
import pandas as pd 
import random
import gzip
import numpy as np
from pandas import read_csv
import time

int2text = {
  0: 'tshirt',
  1: 'trouser',
  2: 'pullover',
  3: 'dress',
  4: 'coat',
  5: 'sandal',
  6: 'shirt',
  7: 'sneaker',
  8: 'bag',
  9: 'ankle_boot',
}
textual_labels = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']


def load_mnist(path, kind='train'):

  """Load MNIST data from `path`"""
  labels_path = os.path.join(path,
                              '%s-labels-idx1-ubyte.gz'
                              % kind)
  images_path = os.path.join(path,
                              '%s-images-idx3-ubyte.gz'
                              % kind)

  with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                              offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                              offset=16).reshape(len(labels), 784)

  return images, labels

def main():

  class_num = 10
  seen_class_num = 5
  unseen_class_num = 5
  seen_samples_per_class = 1200
  data_path = './data/'
  train_file = 'm_train.csv'
  stream_file = 'm_stream.csv'
  # train_file = 'fm_train.csv'
  # stream_file = 'fm_stream.csv'
  # train_file = 'cifar10_train.csv'
  # stream_file = 'cifar10_stream.csv'

  ## ========================================
  # == Get MNIST dataset ====================
  path = './build_dataset/data/mnist/'
  train_data = read_csv(os.path.join(path, "train.csv"), sep=',', header=None).values
  test_data = read_csv(os.path.join(path, "test.csv"), sep=',', header=None).values

  X_train, y_train = train_data[:, 1:], train_data[:, 0]
  X_test, y_test = test_data[:, 1:], test_data[:, 0]

  # print(X_train.shape)
  # print(y_train.shape)
  # print(X_test.shape)
  # print(y_test.shape)
  # time.sleep(5)
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  # path = './build_dataset/data/fmnist/'
  # X_train, y_train = load_mnist(path, kind='train') #(60000, 784), (60000,)
  # X_test, y_test = load_mnist(path, kind='t10k')    #(10000, 784), (10000,)
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Cifar10 dataset ==================
  # data_view = (3, 32, 32)
  # path = './build_dataset/data/cifar10/'
  # train_dataset = read_csv(os.path.join(path, 'cifar10_train.csv'), sep=',', header=None).values
  # test_dataset = read_csv(os.path.join(path, 'cifar10_test.csv'), sep=',', header=None).values

  # X_train = train_dataset[:, :-1]
  # y_train = train_dataset[:, -1]
  # X_test = test_dataset[:, :-1]
  # y_test = test_dataset[:, -1]

  ## ========================================
  ## ========================================

  ## ========================================
  # == For normal training ==================
  # train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)   #(60000, 785)
  # test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)   #(10000, 785)

  # pd.DataFrame(train_data).to_csv(os.path.join(data_path,'fm_train_batch.csv'),
  #   header=None,
  #   index=None
  # )
  # pd.DataFrame(test_data).to_csv(os.path.join(data_path,'fm_test_batch.csv'),
  #   header=None,
  #   index=None
  # )
  # print('done')
  # time.sleep(20)
  ## ========================================
  ## ========================================


  data = np.concatenate((X_train, X_test), axis=0)  #(70000, 784)
  labels = np.concatenate((y_train, y_test), axis=0)#(70000,)

  # convert to textual labels
  # labels = np.array([int2text[int_label] for int_label in labels])

  # == split data by classes =================
  # {label: list[],
  #   ...}

  class_data = {}
  for class_label in set(labels):
    class_data[class_label] = []
  
  for idx, sample in enumerate(data):
    class_data[labels[idx]].append(sample)

  # for class_label in set(labels):
  #   print('{}: {}'.format(class_label, len(class_data[class_label])))
  # time.sleep(4)

  # == Select seen & unseen classes ==========
  seen_class = np.random.choice(class_num, seen_class_num, replace=False)
  unseen_class = [x for x in list(set(labels)) if x not in seen_class]
  # seen_class = [0, 1, 2, 3, 4] 
  # unseen_class = [5, 6, 7, 8, 9]
  # seen_class = np.random.choice(textual_labels, seen_class_num, replace=False)
  # unseen_class = [x for x in list(set(labels)) if x not in seen_class]
  
  print('seen: {}'.format(seen_class))
  print('unseen: {}'.format(unseen_class))
  # time.sleep(2)

  # == Preparing train dataset and test seen data ===
  train_data = []
  test_data_seen = []
  for seen_class_item in seen_class:
    seen_data = np.array(class_data[seen_class_item])

    del class_data[seen_class_item]

    np.random.shuffle(seen_data)
    last_idx = seen_samples_per_class
    test_data_length = seen_data.shape[0] - last_idx
    train_part = seen_data[:last_idx]
    test_part = seen_data[last_idx:]

    train_data_class = np.concatenate((train_part, np.full((last_idx , 1), seen_class_item)), axis=1)
    train_data.extend(train_data_class)

    test_data_class = np.concatenate((test_part, np.full((test_data_length , 1), seen_class_item)), axis=1)
    test_data_seen.extend(test_data_class)
  
  train_data = np.array(train_data) #(6000, 785)
  np.random.shuffle(train_data)
  
  pd.DataFrame(train_data).to_csv(os.path.join(data_path,train_file),
    header=None,
    index=None
  )

  test_data_seen = np.array(test_data_seen) #(30000, 785)
  # == Preparing test(stream) dataset ================
  all_temp_data = []
  add_class_point = 6000
  
  np.random.shuffle(test_data_seen)
  all_temp_data = test_data_seen[:add_class_point]
  test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

  while True:

    if len(unseen_class) != 0:
      rnd_uns_class = unseen_class[0]
      # rnd_uns_class =  random.choice(unseen_class)
      unseen_class.remove(rnd_uns_class)
      
      selected_data = np.array(class_data[rnd_uns_class])
      del class_data[rnd_uns_class]
      temp_data_with_label = np.concatenate((selected_data, np.full((selected_data.shape[0] , 1), rnd_uns_class)), axis=1)
      test_data_seen = np.concatenate((test_data_seen, temp_data_with_label), axis=0)

    np.random.shuffle(test_data_seen)
    all_temp_data = np.concatenate((all_temp_data, test_data_seen[:add_class_point]), axis=0)
    # all_temp_data.append(test_data_seen[:add_class_point])
    test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

    if len(unseen_class) == 0:
      np.random.shuffle(test_data_seen)
      # sections = episode_num - i - 1
      # all_temp_data.extend(test_data_seen)
      all_temp_data = np.concatenate((all_temp_data, test_data_seen), axis=0)
      break
  
  test_data = np.stack(all_temp_data)
  pd.DataFrame(test_data).to_csv(os.path.join(data_path,stream_file), header=None, index=None)

  # dataset = np.concatenate((train_data, test_data), axis=0)
  # file_path = './dataset/fashion-mnist_stream.csv'
  # pd.DataFrame(dataset).to_csv(file_path, header=None, index=None)

if __name__ == '__main__':
  main()