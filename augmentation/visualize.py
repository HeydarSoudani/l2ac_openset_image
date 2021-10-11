import numpy as np
import matplotlib.pyplot as plt
import torchvision

def imshow(imgs):
  img = torchvision.utils.make_grid(imgs)
  img = img / 2 + 0.5     # unnormalize
  npimg = img.detach().cpu().numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()



