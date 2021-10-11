import torch
import torch.nn as nn
import torch.nn.functional as F
import time

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

def euclidean_dist(x, y):
  '''
  Compute euclidean distance between two tensors
  '''
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)



## prototype loss (PL): "Robust Classification with Convolutional Prototype Learning"
class PrototypeLoss(nn.Module):
  def __init__(self):
    super().__init__()
    # self.weights = weights

  def forward(self, features, labels, prototypes):
    # features = features.to('cpu')
    # prototypes = prototypes.to('cpu')
    # labels = labels.to('cpu')
    n = features.shape[0]

    seen_labels = torch.unique(labels)
    prototype_dic = {l.item(): prototypes[idx].reshape(1, -1) for idx, l in enumerate(seen_labels)}
    # print(prototype_dic)
    loss = 0.
    for idx, feature in enumerate(features):
      dists = euclidean_dist(feature.reshape(1, -1), prototype_dic[labels[idx].item()])      #[q_num, cls_num]
      loss += dists
    
    loss /= n
    return loss

class PairwiseLoss(nn.Module):
  def __init__(self, tao=1.0, b=1.0, beta=0.1):
    super().__init__()
    self.b = b
    self.tao = tao
    self.beta = beta

  def forward(self, features, labels, prototypes):
    q_num = features.shape[0]
    cls_num = prototypes.shape[0]

    dists = torch.cdist(features, prototypes)      #[q_num, cls_num]
    likes = torch.ones(q_num, cls_num).to('cuda:0')
    likes[torch.arange(q_num), labels] =  torch.ones(q_num).to('cuda:0')
    inputs = (self.b - likes*(self.tao - dists.pow(2))).flatten()
    
    pw_loss = torch.mean(torch.tensor([self._g(input) for input in inputs]))
    return pw_loss

  def _g(self, z):
    return (1 + (self.beta * z).exp()).log() / self.beta if z < 10.0 else z
    # return (1 + (self.beta * z).exp()).log() / self.beta

class DCELoss(nn.Module):
  def __init__(self, gamma=0.05):
    super().__init__()
    # self.weights = weights
    self.gamma = gamma

  def forward(self, features, labels, prototypes, args):
    features = features.to('cpu')
    prototypes = prototypes.to('cpu')
    n_classes = args.ways
    n_query = args.query_num

    dists = euclidean_dist(features, prototypes)
    dists = (-self.gamma * dists).exp() 

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    # print(log_p_y.shape)
    # print(target_inds.shape)
    # time.sleep(2)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    return loss_val

    # q_num = features.shape[0]
    # dists = torch.cdist(features, prototypes)      #[q_num, ways]
    # exp_dists = (-self.gamma * dists.pow(2)).exp() #[q_num, ways]
    # # exp_dists = self.weights*exp_dists
    # numerator = exp_dists.gather(1, labels.view(-1,1))            #[q_num, 1]
    # denominator = torch.sum(exp_dists, 1).reshape(-1, 1).float()  #[q_num, 1]
    # prob = torch.div(numerator, denominator)                      #[q_num, 1]
    # loss = -prob.log()
    # return loss.sum() / q_num

class CPELoss(nn.Module):
  def __init__(self, args, gamma=0.2, tao=10.0, b=5.0, beta=1.0, lambda_=0.001):
    super().__init__()
    self.lambda_ = lambda_
    self.args = args
    ## with weights
    # weights = torch.tensor(weights).to('cuda:0')
    # self.weights = weights/weights.sum()
    # self.dce = DCELoss(gamma=gamma, weights=self.weights)
    # self.pairwise = PairwiseLoss(tao=tao, b=b, beta=beta)
    # self.proto = PrototypeLoss(weights=self.weights)
    # self.ce = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    
    self.dce = DCELoss(gamma=gamma)
    self.pairwise = PairwiseLoss(tao=tao, b=b, beta=beta)
    self.proto = PrototypeLoss()
    self.ce = torch.nn.CrossEntropyLoss()

  def forward(self, features, outputs, labels, prototypes):
    dce_loss = self.dce(features, labels, prototypes, self.args)
    # cls_loss = F.nll_loss(outputs, labels)
    cls_loss = self.ce(outputs, labels)
    # pairwise_loss = self.pairwise(features, labels, prototypes)
    prototype_loss = self.proto(features, labels, prototypes)

    # return cls_loss + self.lambda_ * dce_loss
    return cls_loss + dce_loss + self.lambda_ * prototype_loss

  def update(self, gamma=0.1, tao=10.0, b=1.0, beta=1.0):
    self.dce.gamma = gamma
    self.pairwise.tao = tao
    self.pairwise.b = b
    self.pairwise.beta = beta



class W_MSE(nn.Module):
  def __init__(self):
    super().__init__()
    
  def __call__(self, output, target, weight=None):
    if weight != None:
      return torch.mean(weight * ((output-target)**2))
    else:
      return torch.mean((output-target)**2)


class W_BCE(nn.Module):
  def __init__(self):
    super().__init__()
    
  def __call__(self, output, target, weight=None):
    if weight != None:
      loss = weight*(- target * torch.log(output) - (1 - target)*torch.log(1 - output))
      return torch.mean(loss)
    else:
      loss = - target * torch.log(output) - (1 - target)*torch.log(1 - output)
      return torch.mean(loss)












class PrototypicalLoss(nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    
    # print('classes: {}'.format(classes))

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    # print('prototypes: {}'.format(prototypes.shape))
    
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    # print('dists: {}'.format(dists.shape))

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    return loss_val

