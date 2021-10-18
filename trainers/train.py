import torch
from torch.nn import BCELoss, MSELoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os
import time

from utils.preparation import dataloader_preparation, relation_data_preparation
from losses import W_MSE, W_BCE


def evaluate(model, mclassifer, dataloader, criterion, args, device):
  with torch.no_grad():
    total_loss = 0.0
    model.eval()
    mclassifer.eval()

    for i, batch in enumerate(dataloader):

      relation_pairs,\
      relarion_labels,\
      relarion_weights = relation_data_preparation(batch, model, args, device)
    
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
          relation_pairs,\
          relarion_labels,\
          relarion_weights = relation_data_preparation(batch, model, args, device)

          ## == relation Net. ==========================
          relations = mclassifer(relation_pairs)
      
          loss = criterion(relations, relarion_labels, weight=relarion_weights)
          model_optim.zero_grad()
          mclassifer_optim.zero_grad()
          loss.backward()
          
          torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
          torch.nn.utils.clip_grad_norm_(mclassifer.parameters(),0.5)
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



  
