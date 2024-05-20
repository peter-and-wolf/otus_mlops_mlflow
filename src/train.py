from typing import Callable

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

import mlflow

from model import MNISTClassifier, MNISTDatasetCsv
from eval import eval


def train(train_path: str, 
          test_path: str,
          model_path: str,
          loss_function: str,
          metric_function: str,
          optimizer_class: str,
          epochs: int, 
          batch_size: int, 
          lr: float):
  
  loss_fn = getattr(nn, loss_function)()
  metric_fn = getattr(torchmetrics, metric_function)(task='multiclass', num_classes=10)
  model = MNISTClassifier()
  optimizer = getattr(torch.optim, optimizer_class)(model.parameters(), lr=lr)

  train_dataloader = DataLoader(
    MNISTDatasetCsv(train_path), 
    batch_size=batch_size, 
    shuffle=True
  )

  test_dataloader = DataLoader(
    MNISTDatasetCsv(test_path), 
    batch_size=batch_size, 
    shuffle=True
  )
  
  step = 0
  best_loss = float('inf')
  num_batches = len(train_dataloader)-1

  for e in range(epochs):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
   
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metric_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric('loss', loss, step=step)
            mlflow.log_metric('accuracy', accuracy, step=step)
            print(f'{step} loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {num_batches}]')
            step += 1
    
    if batch > current:
      print(f'{step} loss: {loss.item():2f} accuracy: {accuracy:2f} [{batch} / {num_batches}]\n')
      mlflow.log_metric('loss', loss, step=step)
      mlflow.log_metric('accuracy', accuracy, step=step)
      step += 1

    eval_loss, eval_accuracy = eval(
       test_dataloader=test_dataloader,
       loss_fn=loss_fn,
       metric_fn=metric_fn,
       model=model,
       epoch=e
    )

    mlflow.log_metric('eval_loss', eval_loss, step=e)
    mlflow.log_metric('eval_accuracy', eval_accuracy, step=e)

    print(f'Eval metrics: \n Accuracy {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n')

    if eval_loss < best_loss:
      print(f'save model with loss {eval_loss}\n')
      torch.save(model.state_dict(), model_path)
      best_loss = eval_loss

