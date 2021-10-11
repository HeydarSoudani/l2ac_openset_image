import torch

from augmentation.operators import traditional_aug, TIM_A, TIM_S
from augmentation.visualize import imshow
import torchvision.transforms.functional as TF

# 1) Traditional augmentation: 
#   Rotate,
#   Random Crop,
#   Color Jittering,
#   Flip,
#   Random Erasing,
#   add Gaussian Noise
# 2) DSTIM consisting of "TIMsub, TIMadd":
#   - "Unsupervised Few-shot Learning via Distribution Shift-based Augmentation, ICCV 2019"
# 3) AutoAugment: 
#   - "Learning Augmentation Strategies from Data, CVPR 2019"

def data_augmentation(support_images,
                      support_labels,
                      query_images,
                      query_labels,
                      args, device):
  """
  Input: 
    - support_images:     [way*shot, 1, 28, 28]
    - support_labels:     [way*shot]
    - query_images:       [way*shot, 1, 28, 28]
    - query_labels:       [way*shot]
  Output: 
    - support_images_aug: [way*shot, 1, 28, 28]
    - support_labels_aug: [way*shot]
    - query_images_aug  : [aug_num*way*shot, 1, 28, 28]
    - query_labels_aug  : [aug_num*way*shot]
  """
  aug_support_images, aug_query_images = traditional_aug(support_images, query_images)
  aug_support_images, aug_query_images = aug_support_images.to(device), aug_query_images.to(device)
  aug_support_labels = support_labels
  aug_query_labels = query_labels
  # support_images = TIM_S(support_images, 0.8, device)
  # query_images = TIM_A(query_images, 0.8, device)

  return (
    aug_support_images,
    aug_support_labels,
    aug_query_images,
    aug_query_labels
  )

