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

      relarion_labels = torch.zeros(args.ways*args.query_num, args.ways).to(device)
      relarion_labels = torch.where(
        support_labels!=query_labels,
        relarion_labels,
        torch.tensor(1.).to(device)
      ).view(-1,1)

      relarion_weights = torch.zeros(args.ways*args.query_num, args.ways).to(device)
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


      ## == relation Net. ==========================
      relations = mclassifer(relation_pairs)
      # loss = criterion(relations, relarion_labels)
      loss = criterion(relations, relarion_labels, weight=relarion_weights)
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
  if args.loss_func == 'mse':
    criterion = W_MSE()
  elif args.loss_func == 'bce':
    criterion = W_BCE()
  
  model_optim = Adam(model.parameters(),lr=args.lr)
  model_scheduler = StepLR(model_optim,step_size=1,gamma=0.5)
  mclassifer_optim = Adam(mclassifer.parameters(),lr=args.lr)
  mclassifer_scheduler = StepLR(mclassifer_optim,step_size=1,gamma=0.5)

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


          ### === For 3-dim feature vector ===========================
          # # each batch sample link to every samples to calculate relations
          # support_features = support_features.view(args.ways, args.shot, 64, 5, 5)
          # support_features = torch.mean(support_features, 1).squeeze(1)
          # support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1,1,1) 
          # support_labels = support_labels[:, 0]
          # support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1)

          # query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1,1,1)
          # query_features_ext = torch.transpose(query_features_ext,0,1)
          # query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)
          # query_labels = torch.transpose(query_labels,0,1)

          # # abssub() cat sum() 
          # sum_feature = support_features_ext+query_features_ext
          # sub_abs_feature = torch.abs(support_features_ext-query_features_ext)
          # relation_pairs = torch.cat((sum_feature, sub_abs_feature), 2).view(-1,64*2,5,5)
          # # cat
          # # relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1,64*2,5,5)
          
          # # n = support_labels.shape[0]
          # # relarion_labels = torch.tensor(
          # #   [1 if support_labels[i] == query_labels[i] else 0 for i in range(n)],
          # #   dtype=torch.float).to(device)
          # # # relarion_labels = torch.zeros(args.ways*args.query_num, args.ways).to(device).scatter_(1, query_labels.view(-1,1), 1.0)
          #  # relarion_weight = torch.tensor(
          # #   [5. if support_labels[i] == query_labels[i] else 1. for i in range(n)],
          # #   dtype=torch.float).to(device)
          # relarion_labels = torch.zeros(args.ways*args.query_num, args.ways).to(device)
          # relarion_labels = torch.where(
          #   support_labels!=query_labels,
          #   relarion_labels,
          #   torch.tensor(1.).to(device)
          # )  

          ### === For 1-dim feature vector =============
          support_features = support_features.view(args.ways, args.shot, 128)
          support_features = torch.mean(support_features, 1).squeeze(1)  # [ways, 128]
          # print(support_features.shape)
          support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num,1,1) #[w*q, w, 128]
          # print(support_features_ext.shape)
          support_labels = support_labels[:, 0]                                           #[w]
          support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num,1) #[w*q, w]

          query_features_ext = query_features.unsqueeze(0).repeat(args.ways,1,1)  #[w, w*q, 128]
          query_features_ext = torch.transpose(query_features_ext,0,1)                #[w*q, w, 128]
          # print(query_features_ext.shape)
          query_labels = query_labels.unsqueeze(0).repeat(args.ways,1)                #[w, w*q]
          query_labels = torch.transpose(query_labels,0,1)                            #[w*q, w]

          # cat
          relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1,128*2) #[w*w*q, 256]

          # abssub() cat sum()
          # sum_feature = support_features_ext+query_features_ext
          # sub_abs_feature = torch.abs(support_features_ext-query_features_ext)
          # relation_pairs = torch.cat((sum_feature, sub_abs_feature), 2).view(-1,128*2) #[w*w*q, 256]

          relarion_labels = torch.zeros(args.ways*args.query_num, args.ways).to(device)
          relarion_labels = torch.where(
            support_labels!=query_labels,
            relarion_labels,
            torch.tensor(1.0).to(device)
          ).view(-1,1)

          relarion_weights = torch.zeros(args.ways*args.query_num, args.ways).to(device)
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

          # print(relarion_weights.view(-1,args.ways))

          ## =================
          ## == relation Net. ==========================
          relations = mclassifer(relation_pairs)
          # print(torch.log(relations.view(-1,args.ways)))
          # print(torch.log(1 - relations.view(-1,args.ways)))
          # print(relarion_labels.view(-1,args.ways))
          # time.sleep(3)
          # loss = criterion(relations, relarion_labels)
          loss = criterion(relations, relarion_labels, weight=relarion_weights)
          model_optim.zero_grad()
          mclassifer_optim.zero_grad()
          loss.backward()
          
          # torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
          # torch.nn.utils.clip_grad_norm_(mclassifer.parameters(),0.5)
          model_optim.step()
          mclassifer_optim.step()

          train_loss += loss

          ## == validation ==============
          if (miteration_item + 1) % args.log_interval == 0:
            train_loss_total = train_loss / args.log_interval
            train_loss = 0.0

            val_loss_total = evaluate(model, mclassifer, val_dataloader, criterion, args, device)

            # print losses
            print('Time: %f, Step: %d, Train Loss: %.9f, Val Loss: %.9f' % (
              time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
            print('===============================================')
            global_time = time.time()
      
            # save best model
            if val_loss_total < min_loss:
              model.save(os.path.join(args.save, "model_best.pt"))
              mclassifer.save(os.path.join(args.save, "mclassifier_best.pt"))
              min_loss = val_loss_total
              print("Saving new best model")

        model_scheduler.step()
        mclassifer_scheduler.step()

  except KeyboardInterrupt:
    print('skipping training') 
  
  # save last model
  model.save(os.path.join(args.save, "model_last.pt"))
  mclassifer.save(os.path.join(args.save, "mclassifier_last.pt"))
  print("Saving new last model")





if __name__ == '__main__':
  pass



  
