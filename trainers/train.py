import torch
from torch.nn import BCELoss, MSELoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
import time

from utils.dataloader_preparation import dataloader_preparation
from models.relation import RelationMLP
from losses import W_MSE, W_BCE

def evaluate(model, mclassifer, dataloader, criterion, args, device):
  with torch.no_grad():
    total_loss = 0.0
    model.eval()
    mclassifer.eval()

    for i, batch in enumerate(dataloader):

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

      support_features = support_features.view(args.ways, args.shot, 64, 5, 5)
      support_features = torch.sum(support_features, 1).squeeze(1)
      support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1,1,1)
      support_labels = support_labels[:, 0]
      support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1).flatten()

      query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1,1,1)
      query_features_ext = torch.transpose(query_features_ext,0,1)
      query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)
      query_labels = torch.transpose(query_labels,0,1).flatten()

      relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,64*2,5,5)
      n = support_labels.shape[0]
      relarion_labels = torch.tensor(
        [1 if support_labels[i] == query_labels[i] else 0 for i in range(n)],
        dtype=torch.float).to(device)
                          
      relations = mclassifer(relation_pairs).view(-1,args.ways)

      loss = criterion(relations.flatten(), relarion_labels)
      loss = loss.mean()
      total_loss += loss.item()
    
    total_loss /= len(dataloader)
    return total_loss  
    


def train(model,
          mclassifer,
          train_data,
          args,
          device):

  model.to(device)  # this is feature extractor model
  mclassifer.to(device)
  ## == Prepar dataloader =============
  train_dataloaders, val_dataloader=  dataloader_preparation(train_data, args)
  

  # criterion = BCELoss()
  # criterion = MSELoss()
  criterion = W_MSE()
  # criterion = W_BCE()
  
  model_optim = Adam(model.parameters(),lr=args.lr)
  model_scheduler = StepLR(model_optim,step_size=100000,gamma=0.5)
  mclassifer_optim = Adam(mclassifer.parameters(),lr=args.lr)
  mclassifer_scheduler = StepLR(mclassifer_optim,step_size=100000,gamma=0.5)

  ## == Training ======================
  global_time = time.time()
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('===================================== Epoch %d =====================================' % epoch_item)
      train_loss = 0.0
      
      for train_loader in train_dataloaders:
        for miteration_item, batch in enumerate(train_loader):
          model.train()
          mclassifer.train()
          
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

          # each batch sample link to every samples to calculate relations
          support_features = support_features.view(args.ways, args.shot, 64, 5, 5)
          support_features = torch.sum(support_features, 1).squeeze(1)
          support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1,1,1) 
          support_labels = support_labels[:, 0]
          support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1).flatten()

          query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1,1,1)
          query_features_ext = torch.transpose(query_features_ext,0,1)
          query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)
          query_labels = torch.transpose(query_labels,0,1).flatten()

          relation_pairs = torch.cat((support_features_ext,query_features_ext),2).view(-1,64*2,5,5)
          n = support_labels.shape[0]
          relarion_labels = torch.tensor(
            [1 if support_labels[i] == query_labels[i] else 0 for i in range(n)],
            dtype=torch.float).to(device)
          relarion_weight = torch.tensor(
            [5. if support_labels[i] == query_labels[i] else 1. for i in range(n)],
            dtype=torch.float).to(device)
         
          # print(relation_pairs.shape)  
          # print(relarion_labels)  
          relations = mclassifer(relation_pairs)
          # print(relations.shape)
          loss = criterion(relations.flatten(), relarion_labels, weight=relarion_weight)

          model_optim.zero_grad()
          mclassifer_optim.zero_grad()
          loss.backward()
          # torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
          # torch.nn.utils.clip_grad_norm_(mclassifer.parameters(),0.5)
          model_optim.step()
          model_optim.step()

          train_loss += loss


          ## == validation ==============
          if (miteration_item + 1) % args.log_interval == 0:
            train_loss_total = train_loss / args.log_interval
            train_loss = 0.0

            val_loss_total = evaluate(model, mclassifer, val_dataloader, criterion, args, device)

            # print losses
            print('Time: %f, Step: %d, Train Loss: %f, Val Loss: %f' % (
              time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
            print('===============================================')
            global_time = time.time()
      
            # save best model
            if val_loss_total < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              mclassifer.save(os.path.join(args.save, "mclassifier_best.pt"))
              min_loss = val_loss_total
              print("Saving new best model")

        model_scheduler.step(miteration_item)
        mclassifer_scheduler.step(miteration_item)

  except KeyboardInterrupt:
    print('skipping training') 
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  mclassifer.save(os.path.join(args.save, "mclassifier_last.pt"))
  print("Saving new last model")





if __name__ == '__main__':
  pass



  
