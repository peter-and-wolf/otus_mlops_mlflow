from typing import Callable

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

import mlflow

from model import MNISTClassifier, MNISTDatasetCsv


def eval(test_dataloader: DataLoader, 
         loss_fn: Callable,
         metric_fn: Callable,
         model: MNISTClassifier,
         epoch: int = None) -> tuple[float, float]:
   
  num_batches = len(test_dataloader)

  model.eval()
  eval_loss, eval_metric = .0, .0
  with torch.no_grad():
    for X, y in test_dataloader:
      pred = model(X)
      eval_loss += loss_fn(pred, y).item()
      eval_metric += metric_fn(pred, y)
    
    eval_loss /= num_batches
    eval_metric /= num_batches

  return eval_loss, eval_metric
  

def test(test_path: str,
         model_path: str,
         batch_size: int,
         loss_function: str,
         metric_function: str):
   
  loss_fn = getattr(nn, loss_function)()
  metric_fn = getattr(torchmetrics, metric_function)(task='multiclass', num_classes=10)

  test_dataloader = DataLoader(
    MNISTDatasetCsv(test_path), 
    batch_size=batch_size, 
    shuffle=True
  )

  model = MNISTClassifier()
  model.load_state_dict(torch.load(model_path))
  
  best_loss, best_accuracy = eval(
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    metric_fn=metric_fn,
    model=model
  )

  mlflow.log_metric('best_loss', best_loss)
  mlflow.log_metric('best_accuracy', best_accuracy)
  mlflow.pytorch.log_model(model, 'model')

  print(f'Best metrics: \n Accuracy {best_accuracy:.2f}, Best Avg loss: {best_loss:2f} \n')
  