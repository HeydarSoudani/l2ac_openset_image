
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from pandas import read_csv

from models.cnn import CNNEncoder, CNNEncoder_2
from models.densenet import DenseNet
from augmentation import transforms
from datasets.dataset import DatasetFM


def plot_tsne(args, device):
  
  ## == Load data ==============
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  stream_data = read_csv(args.test_path, sep=',', header=None).values
  stream_dataset = DatasetFM(stream_data)
  # stream_dataset = DatasetFM(stream_data, transforms=transform)
 
  ## == Load model ==============
  # model = CNNEncoder(args)
  model = CNNEncoder_2(args)
  # model = DenseNet(args, tensor_view=(3, 32, 32))
  if args.which_model == 'best':
    try:
      model.load(args.best_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.best_model_path))
  elif args.which_model == 'last':
    try:
      model.load(args.last_model_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.last_model_path))
  # model.to(device)
  
  ### ======================================
  ### == Feature space visualization =======
  ### ======================================
  print('=== Feature-Space visualization (t-SNE) ===')
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)

  with torch.no_grad():
    batches, batch_labels = stream_dataset.get_sample_per_class(n_samples=600)
    # batches, batch_labels = batches.to(device), batch_labels.to(device)
    out, feature = model.forward(batches)
    feature = feature.cpu().detach().numpy()
    # batches = batches.view(batches.size(0), -1)
    # batches = batches.cpu().detach().numpy()
    batch_labels = batch_labels.cpu().detach().numpy()

  tsne = TSNE()
  X_embedded = tsne.fit_transform(feature)
  sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=batch_labels, legend='full', palette=palette)
  plt.show()
  ### ======================================
