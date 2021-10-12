import torch
from torch.utils.data import DataLoader
from pandas import read_csv
import time

from detectors.detector import Ranker, Detector

from utils.data_selector import DataSelector
from datasets.dataset import DatasetFM
from augmentation import transforms
from trainers.train import train


def stream_learn(model, mclassifer, args,  device, known_labels=None):
  args.epochs = args.retrain_epochs
  args.meta_iteration = args.retrain_meta_iteration
  
  print('================================ Stream learning L2AC ===========================')
  # == Data ==================================
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  stream_data = read_csv(args.test_path, sep=',', header=None).values
  stream_dataset = DatasetFM(stream_data)
  # stream_dataset = DatasetFM(stream_data, transforms=transform)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

  train_data = read_csv(args.train_path, sep=',', header=None).values
  train_dataset = DatasetFM(train_data)
  base_labels = train_dataset.label_set
  

  ## == Load Model & MetaClassifier ====================
  if args.which_model == 'best':
    try:
      model.load(args.best_model_path)
      mclassifer.load(args.best_mclassifier_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.best_model_path))
      print("Load meta-classifier from file {}".format(args.best_mclassifier_path))
  elif args.which_model == 'last':
    try:
      model.load(args.last_model_path)
      mclassifer.load(args.last_mclassifier_path)
    except FileNotFoundError:
      pass
    else:
      print("Load model from file {}".format(args.last_model_path))
      print("Load meta-classifier from file {}".format(args.last_mclassifier_path))
  model.to(device)
  mclassifer.to(device)


  ## == Create ranker ============================
  ranker = Ranker(train_data, model, device, k=10)
  detector = Detector(base_labels)

  ## == Selector =================================
  retrain_data_selector = DataSelector(
                            init_data=train_data,
                            model=model,
                            device=device)

  buffer = [] 
  ## == Stream ===================================
  for i, data in enumerate(dataloader):
    model.eval()
    mclassifer.eval()
    
    sample, label = data
    sample, label = sample.to(device), label.to(device)

    with torch.no_grad():
      feature = model.forward(sample)
    topk_data, topk_label = ranker.topk_selection(feature)
    xt_repeat = feature.repeat(args.ways*10, 1)  #[50, 128]
    relation_input = torch.cat((xt_repeat, topk_data), axis=1)
    relation_output = mclassifer(relation_input)

    detected_novelty, predicted_label, prob = detector(relation_output, topk_label)
    real_novelty = label.item() not in detector.base_labels

    if detected_novelty:
      sample = torch.squeeze(sample, 0)
      buffer.append((sample, label))

    print("[stream %5d]: %d, %d, %7.4f, %5s, %5s"%
      (i+1, label, predicted_label, prob, real_novelty, detected_novelty))

    if len(buffer) == args.buffer_size:
      new_train_data = retrain_data_selector.renew(buffer)
      train(model, new_train_data, args, device)
      ranker.update_memory(new_train_data, model)
      
      time.sleep(3)
      buffer.clear()



