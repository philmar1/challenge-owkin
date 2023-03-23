from .model import *
from .utils import *

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch import nn, optim

from tqdm import tqdm

from typing import Dict
import time 
import logging

logger = logging.getLogger(__name__)

def get_lr_scheduler(optimizer, warmup, max_iters) -> CosineWarmupScheduler:
    #return CosineWarmupScheduler(optimizer, warmup, max_iters)
    p = nn.Parameter(torch.empty(4, 4))
    opt = optim.Adam([p], lr=1e-3)
    return optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.65 ** epoch)

def weighted_loss(weight: float):
    bce_loss = nn.BCELoss(reduction='none')
    def loss(y_pred, y_true):
        intermediate_loss = bce_loss(y_pred, y_true)
        return torch.mean(weight * y_true * intermediate_loss + (1 - y_true) * intermediate_loss)
    return loss

def warmup_lr(optimizer, start_lr: float, end_lr: float, epoch_max: int)        :
    step_lr = (end_lr - start_lr)/epoch_max
    lr = optimizer.param_groups[0]['lr']
    lr += step_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #return optimizer

def set_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #return optimizer

def get_optimizer(model, lr: float = 0.000001):
    return optim.Adam(model.parameters(), lr=lr)

def train_epoch(model, optimizer, lr_scheduler, loss_function, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    progress = tqdm(enumerate(dataloader), desc="Training Loss: ", total=len(dataloader))
    
    total_loss = 0
    model.train()
    for i, data in progress: 
        lr = optimizer.param_groups[0]['lr']
        
        X, y = data[0].to(device), data[1].to(device)
        y = y.reshape(-1,1)
            
        # training step for single batch
        model.zero_grad() 
            
        outputs = model(X) 
        loss = loss_function(outputs, y) 
        loss.backward() 
        optimizer.step() 
        #lr_scheduler.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Training Loss: {:.4f}, learning_rate : {:.7f}".format(total_loss/(i+1), lr))
        
    return model, total_loss/(i+1) #current_loss    
    
def eval_epoch(model, loss_function, dataloader):
    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    y_pred, y_true = [], []
    precision, recall, f1, accuracy = [], [], [], []
        
    # set model to evaluating (testing)
    model.eval()
    progress = tqdm(enumerate(dataloader), desc="Validation Loss: ", total=len(dataloader))
    with torch.no_grad():
        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            y = y.reshape(-1,1)
            outputs = model(X) 
            prediced_classes = outputs.detach().round()
            total_loss += loss_function(outputs, y)
            y_pred.extend(prediced_classes.reshape(-1).tolist())
            y_true.extend(y.reshape(-1).tolist())
                    
            # updating progress bar
            progress.set_description("Validation Loss: {:.4f}".format(total_loss/(i+1)))
        # calculate P/R/F1/A metrics for batch
        for acc, metric in zip((precision, recall, f1, accuracy), 
                                (precision_score, recall_score, f1_score, accuracy_score)):
            acc.append(metric(y_true, y_pred))
                
    return total_loss/i, precision[0], recall[0], f1[0], accuracy[0]
    
def train(model, train_loader, eval_loader, hyperparameters: Dict):
    # Get training parameters
    epochs = hyperparameters['epochs']
    positive_weight = hyperparameters['positive_weight']
    gamma = hyperparameters['gamma']
    warmup_epoch = hyperparameters['warmup_epoch']
    end_lr = hyperparameters['end_lr']
    start_lr = hyperparameters['start_lr']
    
    optimizer = get_optimizer(model, lr=start_lr)
    lr_scheduler = get_lr_scheduler(optimizer, warmup_epoch, max_iters=1000)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_function = nn.BCELoss(reduction='mean') #weighted_loss(weight=positive_weight)
    
    train_batches = len(train_loader)
    eval_batches = len(eval_loader)
    start_ts = time.time()
    
    losses_train, losses_val = [], []
    logger.info("Starting training with lr: {:.7f}".format(start_lr))
    for epoch in range(epochs):
        logger.info("Epoch {}/{}".format(epoch + 1, epochs))
        lr = optimizer.param_groups[0]['lr']
        
        if 0 < epoch <= warmup_epoch:
            warmup_lr(optimizer, start_lr, end_lr, warmup_epoch)
        if epoch > warmup_epoch:
            set_lr(optimizer, lr * gamma)
        
        model, train_loss = train_epoch(model, optimizer, lr_scheduler, loss_function, train_loader)
        val_loss, precision, recall, f1, accuracy = eval_epoch(model, loss_function, eval_loader)
        
        losses_train.append(train_loss/train_batches) # for plotting learning curve
        losses_val.append(val_loss/train_batches) # for plotting learning curve
        
        print(f"Epoch {epoch + 1}/{epochs}, lr {lr:.10f}, training loss: {train_loss/train_batches}, validation loss: {val_loss/eval_batches}")
        print_scores(precision, recall, f1, accuracy)
        
        
    logger.info(f"Training time: {time.time()-start_ts}s")

    return model

