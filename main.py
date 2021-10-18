import os
import time
import torch
import argparse
import numpy as np
from pandas import read_csv

# CNNEncoder, DenseNet,
from models.cnn import CNNEncoder, weights_init
from models.relation import RelationNetwork, RelationNetworkFC

from init_learn import init_learn
from zeroshot_test import zeroshot_test
from stream_learn import stream_learn
# from utils.plot_tsne import plot_tsne
# from utils.functions import set_novel_label, mapping_text2int


## == Params ===========
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='init_learn', help='')
parser.add_argument('--loss_func', type=str, default='mse', help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--retrain_epochs', type=int, default=1, help='')
parser.add_argument('--meta_iteration', type=int, default=5000, help='')
parser.add_argument('--retrain_meta_iteration', type=int, default=1000, help='')
parser.add_argument('--log_interval', type=int, default=200, help='must be less then meta_iteration parameter')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=1, help='')
parser.add_argument('--query_num', type=int, default=5, help='')
parser.add_argument('--buffer_size', type=int, default=1000, help='')
parser.add_argument('--update_step', type=int, default=5, help='for Reptile algorithm')

parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=0.0005, help='')  #l2 regularization
parser.add_argument('--gamma', type=float, default=0.1, help='for lr step')
parser.add_argument('--beta', type=float, default=0.06, help='for Reptile algorithm')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--std_coefficient', type=float, default=1.0, help='for Prototype algorithm')

parser.add_argument('--which_model', type=str, default='best', help='')
parser.add_argument('--dropout', type=float, default=0.4, help='')
parser.add_argument('--hidden_dims', type=int, default=128, help='') #768
parser.add_argument('--seen_labels', type=int, default=5, help='')

parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--train_path', type=str, default='data/fm_train.csv', help='')
parser.add_argument('--test_path', type=str, default='data/fm_stream.csv', help='')
# parser.add_argument('--train_path', type=str, default='data/cifar10_train_batch.csv', help='')
# parser.add_argument('--test_path', type=str, default='data/cifar10_test_batch.csv', help='')
parser.add_argument('--best_model_path', type=str, default='saved/model_best.pt', help='')
parser.add_argument('--last_model_path', type=str, default='saved/model_last.pt', help='')
parser.add_argument('--best_mclassifier_path', type=str, default='saved/mclassifier_best.pt', help='for l2ac')
parser.add_argument('--last_mclassifier_path', type=str, default='saved/mclassifier_last.pt', help='for l2ac')

# WideResNet Model
parser.add_argument('--depth', type=int, default=28, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

# Random Erasing
parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.2, type=float, help='aspect of erasing area')

# Network Setting
parser.add_argument('--relation_dim', default=3, type=int, choices=[1,3])
parser.add_argument('--use_transform', default=False, type=bool)
parser.add_argument('--rel_input_oprations', default='cat', type=str, choices=['cat','sum_sub_cat'])


args = parser.parse_args()

## == Device =====================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Device: {}'.format(device))

## == Apply seed =================
torch.manual_seed(args.seed)
np.random.seed(args.seed)

## == Save dir ===================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Models Definition ==========
model = CNNEncoder(args)

if args.relation_dim == 1:
  mclassifer = RelationNetworkFC(args)
elif args.relation_dim == 3:
  mclassifer = RelationNetwork(64, 8)

model.apply(weights_init)
mclassifer.apply(weights_init)


if __name__ == '__main__':
  if args.phase == 'init_learn':
    init_learn(model, mclassifer, args, device)
  elif args.phase == 'zeroshot_test':
    zeroshot_test(model, mclassifer, args, device)
  elif args.phase == 'stream_learn':
    stream_learn(model, mclassifer, args, device)


