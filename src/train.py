from pathlib import Path
from typing import Annotated, Callable


import typer
from tqdm import tqdm as progress

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

from model import MNISTClassifier
from dataset import get_dataloaders
from tracker import ExperimentTracker
from dvc_tracker import start_tracker


def train(
    epochs: int,
    tracker: ExperimentTracker,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: MNISTClassifier,
    loss_fn: Callable,
    metric_fn: Callable,
    optimizer: torch.optim.Optimizer,
    model_path: Annotated[Path, typer.Option()] = Path('data/model.pt'),
  ) -> None:
  
    train_step, val_step = 1, 1

    for _ in range(epochs):
      # Training loop 
      for X, y in progress(train_dataloader, total=len(train_dataloader), desc='Training'):
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metric_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tracker.log_metric(f'training/loss', loss, train_step)
        tracker.log_metric(f'training/accuracy', accuracy, train_step)

        train_step += 1

      # Validation loop
      with torch.no_grad():
        for X, y in progress(val_dataloader, total=len(val_dataloader), desc='Validation'):
          pred = model(X)
          loss = loss_fn(pred, y)
          accuracy = metric_fn(pred, y)
          
          tracker.log_metric(f'validation/loss', loss, val_step)
          tracker.log_metric(f'validation/accuracy', accuracy, val_step)

          val_step += 1
  
    # Save model  
    torch.save(model.state_dict(), model_path)


def main(
    train_path: Annotated[Path, typer.Option()] = Path('data/train_data.npy'),
    model_path: Annotated[Path, typer.Option()] = Path('data/model.pt'),
    epochs: Annotated[int, typer.Option()] = 3, 
    batch_size: Annotated[int, typer.Option()] = 32, 
    lr: Annotated[float, typer.Option()] = .01
  ) -> None:

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    model = MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader, val_dataloader = get_dataloaders(train_path, batch_size=batch_size)
   
    with start_tracker() as tracker:
      
      tracker.log_params({
        'loss_function': 'CrossEntropyLoss',
        'metric_function': 'Accuracy',
        'optimizer_class': 'SGD',
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
      })
      
      train(
        epochs=epochs,
        tracker=tracker,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        model_path=model_path
      ) 


if __name__ == '__main__':
  typer.run(main)
