import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentation import transforms
from generator import task_generator
from datasets.dataset import DatasetFM
from sampler import TaskSampler


def dataloader_preparation(train_data, args):
  print(train_data.shape)

  n, _ = train_data.shape
  np.random.shuffle(train_data)
  train_val_data = np.split(train_data, [int(n*0.9), n])
  train_data = train_val_data[0]
  val_data = train_val_data[1]

  ## == 1) Create tasks from train_data
  # task: [n, 765(feature_in_line+label)]
  task_list = task_generator(train_data, args, task_number=1, type='random') #['random', 'dreca']

  ## =========
  transform_train = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    # transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])
  transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  ## = trainloader ============
  train_dataloaders = []
  for task_data in task_list:
    temp_dataset = DatasetFM(task_data)
    # temp_dataset = DatasetFM(task_data, transforms=transform_train)

    train_sampler = TaskSampler(
      temp_dataset,
      n_way=args.ways,
      n_shot=args.shot,
      n_query=args.query_num,
      n_tasks=args.meta_iteration
    )
    train_loader = DataLoader(
      temp_dataset,
      batch_sampler=train_sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=train_sampler.episodic_collate_fn,
    )
    train_dataloaders.append(train_loader)

  ## = Data validation ================
  val_dataset = DatasetFM(val_data)
  # val_dataset = DatasetFM(val_data, transforms=transform_val)
  val_sampler = TaskSampler(
    val_dataset,
    n_way=args.ways,
    n_shot=args.shot,
    n_query=args.query_num,
    n_tasks=100
  )
  val_dataloader = DataLoader(
      temp_dataset,
      batch_sampler=val_sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=val_sampler.episodic_collate_fn,
    )

  # val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

  return train_dataloaders, val_dataloader



def relation_data_preparation(batch, model, args, device):
  # == Data preparation ===========
  support_images, support_labels, query_images, query_labels = batch

  support_images = support_images.reshape(-1, *support_images.shape[2:])
  # support_labels = support_labels.flatten() #[5]
  query_images = query_images.reshape(-1, *query_images.shape[2:])
  query_labels = query_labels.flatten() #[5]
  support_images = support_images.to(device)
  support_labels = support_labels.to(device)
  query_images = query_images.to(device)
  query_labels = query_labels.to(device)
  
  support_features = model.forward(support_images) #[ways*shot, 64, 5, 5]
  query_features = model.forward(query_images)     #[ways*query_num, 64, 5, 5]

  ### === For 3-dim feature vector ============
  # support_features = support_features.view(args.ways, args.shot, 64, 5, 5)
  # support_features = torch.mean(support_features, 1).squeeze(1)
  # support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1,1,1)
  # support_labels = support_labels[:, 0]
  # support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1)

  # query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1,1,1)
  # query_features_ext = torch.transpose(query_features_ext,0,1)
  # query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)
  # query_labels = torch.transpose(query_labels,0,1)

  # # sum, sub, cat
  # sum_feature = support_features_ext+query_features_ext
  # sub_abs_feature = torch.abs(support_features_ext-query_features_ext)
  # relation_pairs = torch.cat((sum_feature, sub_abs_feature), 2).view(-1,64*2,5,5)
  # # cat
  # # relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,64*2,5,5)
  
  # relarion_labels = torch.zeros(args.ways*args.query_num, args.ways).to(device)
  # relarion_labels = torch.where(
  #   support_labels!=query_labels,
  #   relarion_labels,
  #   torch.tensor(1.).to(device)
  # )

  ### === For 1-dim feature vector =============
  support_features = support_features.view(args.ways, args.shot, 128)
  support_features = torch.mean(support_features, 1).squeeze(1)  # [ways, 128]
  support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1) #[w*q, w, 128]
  support_labels = support_labels[:, 0]                                           #[w]
  support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1) #[w*q, w]

  query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1)  #[w, w*q, 128]
  query_features_ext = torch.transpose(query_features_ext,0,1)                #[w*q, w, 128]
  query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)                #[w, w*q]
  query_labels = torch.transpose(query_labels,0,1)                            #[w*q, w]

  # cat
  relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1,128*2)

  # abssub() cat sum() 
  # sum_feature = support_features_ext+query_features_ext
  # sub_abs_feature = torch.abs(support_features_ext-query_features_ext)
  # relation_pairs = torch.cat((sum_feature, sub_abs_feature), 2).view(-1,128*2) #[w*w*q, 256]

  relarion_labels = torch.zeros(
    args.ways*args.query_num,
    args.ways).to(device)
  relarion_labels = torch.where(
    support_labels!=query_labels,
    relarion_labels,
    torch.tensor(1.).to(device)
  ).view(-1,1)

  relarion_weights = torch.zeros(
    args.ways*args.query_num,
    args.ways).to(device)
  relarion_weights = torch.where(
    support_labels!=query_labels,
    relarion_weights,
    torch.tensor(0.83).to(device)
  )
  relarion_weights = torch.where(
    support_labels==query_labels,
    relarion_weights,
    torch.tensor(0.17).to(device)
  ).view(-1,1)

  return relation_pairs, relarion_labels, relarion_weights

