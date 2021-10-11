import numpy as np
from itertools import combinations


# type: ['random', 'way_less_label']
def task_generator(data, args, task_number=2, type='random'):
  
  np.random.shuffle(data)
  task_list = []
  labels = data[:, -1]
  label_set = set(labels)
  
  # = Split by class
  class_data = {}
  for class_label in label_set:
    class_data[class_label] = []
  for idx, sample in enumerate(data):
    class_data[labels[idx]].append(sample)


  if type == 'random':
    for class_label in label_set:
      class_data[class_label] = np.array_split(np.array(class_data[class_label]), task_number)

    for task_idx in range(task_number):
      temp = [class_data[class_label][task_idx]  for class_label in label_set ]
      task_list.append(np.concatenate(temp, axis=0))


  elif type == 'ways_less_seen':
    split_num = len(list(combinations([i for i in range(args.seen_labels-1)], args.ways-1)))
    for class_label in label_set:
      class_data[class_label] = np.array_split(np.array(class_data[class_label]), split_num)
    
    comb = list(combinations([i for i in range(args.seen_labels)], args.ways))
    for idxs in comb:
      temp = []
      for idx in idxs:
        temp.append(class_data[idx][0])
        del class_data[idx][0]
      task_list.append(np.concatenate(temp, axis=0))

  return task_list

