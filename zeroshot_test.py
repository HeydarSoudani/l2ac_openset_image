import torch
from torch.utils.data import DataLoader
from pandas import read_csv

from detectors.detector import Ranker, Detector

from datasets.dataset import DatasetFM
from augmentation import transforms


def zeroshot_test(model, mclassifer, args, device, known_labels=None):
  print('================================ Zero-Shot Test L2AC ===========================')
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
  

  ## == Load model ============================
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

  ## == Create ranker ===========================
  k = 10
  ranker = Ranker(train_data, model, device, k=k)
  detector = Detector(base_labels)

  detection_results = []
  ## == Test Model ==============================
  model.eval()
  mclassifer.eval()
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      sample, label = data
      sample, label = sample.to(device), label.to(device)
      feature = model.forward(sample)
      
      topk_data, topk_label = ranker.topk_selection(feature)
      

      ### === For 3-dim feature vector ===========================
      # xt_repeat = feature.repeat(args.ways*k, 1, 1, 1)  #[50, 64, 5, 5]
      # # sum, sub, cat
      # sum_feature = xt_repeat+topk_data
      # sub_abs_feature = torch.abs(xt_repeat-topk_data)
      # relation_input = torch.cat((sum_feature, sub_abs_feature), 2).view(-1,64*2,5,5)
      # # cat
      # # relation_input = torch.cat((xt_repeat, topk_data), axis=1)
      
      ### === For 1-dim feature vector ===========================
      xt_repeat = feature.repeat(args.ways*k, 1)  #[50, 128]
      # sum, sub, cat
      sum_feature = xt_repeat+topk_data
      sub_abs_feature = torch.abs(xt_repeat-topk_data)
      relation_input = torch.cat((sum_feature, sub_abs_feature), 1).view(-1,128*2) #[w*w*q, 256]
      
      relation_output = mclassifer(relation_input)

      detected_novelty, predicted_label, prob = detector(relation_output, topk_label)
      real_novelty = label.item() not in detector.base_labels
      detection_results.append((label.item(), predicted_label, real_novelty, detected_novelty))

      print("[stream %5d]: %d, %d, %7.4f, %5s, %5s"%
        (i+1, label, predicted_label, prob, real_novelty, detected_novelty))

    tp, fp, fn, tn, cm, acc, acc_all = detector.evaluate(detection_results)
    precision = tp / (tp + fp + 1)
    recall = tp / (tp + fn + 1)
    M_new = fn / (tp + fn + 1)
    F_new = fp / (fp + tn + 1)

    print("true positive: %d"% tp)
    print("false positive: %d"% fp)
    print("false negative: %d"% fn)
    print("true negative: %d"% tn)
    print("precision: %7.4f"% precision)
    print("recall: %7.4f"% recall)
    print("M_new: %7.4f"% M_new)
    print("F_new: %7.4f"% F_new)
    print("Accuracy: %7.4f"% acc)
    print("Accuracy All: %7.4f"% acc_all)
    print("confusion matrix: \n%s"% cm)



