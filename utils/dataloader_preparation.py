import numpy as np
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
    # temp_dataset = DatasetFM(task_data)
    temp_dataset = DatasetFM(task_data, transforms=transform_train)

    train_sampler = TaskSampler(
      temp_dataset,
      n_way=args.ways,
      # n_query_way=args.query_ways,
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
    # n_query_way=args.query_ways,
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