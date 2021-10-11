import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from datasets.dataset import DatasetFM

class Ranker():
  def __init__(self, init_data, model, device, k=5):
    self.k = k
    self.device = device
    self.cos_sim = nn.CosineSimilarity()

    init_dataset = DatasetFM(init_data)
    dataloader = DataLoader(dataset=init_dataset, batch_size=1, shuffle=False)
    self.labels_set = set(init_dataset.labels)

    ## == split data by classes =================
    self.data_class = {}
    for class_label in self.labels_set:
      self.data_class[class_label] = []
    for idx, (sample, label) in enumerate(dataloader):
      
      with torch.no_grad():
        feature = model(sample.to(self.device)) 
      self.data_class[label.item()].append(feature)

    for class_label in self.labels_set:
      self.data_class[class_label] = torch.cat(self.data_class[class_label])

  def topk_selection(self, sample):
    data_selected = []
    labels = []
    for clabel in self.labels_set:

      sim_scores = self.cos_sim(
        sample.view(sample.shape[0], -1),
        self.data_class[clabel].view(self.data_class[clabel].shape[0], -1)
      )
      topk_idx = torch.topk(sim_scores, self.k).indices
      data_selected.append(self.data_class[clabel][topk_idx])
      labels.extend(torch.full((self.k, ), clabel, dtype=torch.float)) #[k, ]
    data = torch.cat(data_selected)
    return data, torch.tensor(labels) #[cls*k, feature_size], [cls*k,]

  def update_memory(self, new_data, new_model):
    new_dataset = DatasetFM(new_data)
    dataloader = DataLoader(dataset=new_dataset, batch_size=1, shuffle=False)
    self.labels_set = set(new_dataset.labels)

    ## == split data by classes =================
    self.data_class = {}
    for class_label in self.labels_set:
      self.data_class[class_label] = []
    for idx, (sample, label) in enumerate(dataloader):
      
      with torch.no_grad():
        _, feature = new_model(sample.to(self.device))
      self.data_class[label.item()].append(feature)

    for class_label in self.labels_set:
      self.data_class[class_label] = torch.cat(self.data_class[class_label])


class Detector(object):
  def __init__(self, label_set=()):
    self.base_labels = label_set
    self._known_labels = self.base_labels
    self.threshold = 0.5

  def __call__(self, relation_output, labels):
    detected_novelty = False
    prob = torch.max(relation_output)
    predicted_label = labels[torch.argmax(relation_output)]

    if prob < self.threshold:
      detected_novelty = True
    
    return detected_novelty, predicted_label, prob

  def evaluate(self, results):
    self.results = np.array(results, dtype=[
      ('true_label', np.int32),
      ('predicted_label', np.int32),
      # ('probability', np.float32),
      # ('distance', np.float32),
      ('real_novelty', np.bool),
      ('detected_novelty', np.bool)
    ])

    real_novelties = self.results[self.results['real_novelty']]
    detected_novelties = self.results[self.results['detected_novelty']]
    detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

    true_positive = len(detected_real_novelties)
    false_positive = len(detected_novelties) - len(detected_real_novelties)
    false_negative = len(real_novelties) - len(detected_real_novelties)
    true_negative = len(self.results) - true_positive - false_positive - false_negative

    cm = confusion_matrix(self.results['true_label'], self.results['predicted_label'], sorted(list(np.unique(self.results['true_label']))))
    results = self.results[np.isin(self.results['true_label'], list(self._known_labels))]
    acc = accuracy_score(results['true_label'], results['predicted_label'])
    acc_all = accuracy_score(self.results['true_label'], self.results['predicted_label'])

    return true_positive, false_positive, false_negative, true_negative, cm, acc, acc_all
