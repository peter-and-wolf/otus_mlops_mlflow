from pathlib import Path
from typing_extensions import Annotated

import typer 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

from model import MNISTClassifier
from dataset import get_dataloaders
from tracker import ExperimentTracker
from mlflow_tracker import start_tracker
from preprocess import preprocess
from train import train
from eval import eval


def main(
    train_csv_path: Annotated[Path, typer.Option()] = Path('data/train_data.csv'),
    test_csv_path: Annotated[Path, typer.Option()] = Path('data/test_data.csv'),
    train_npy_path: Annotated[Path, typer.Option()] = Path('data/train_data.npy'),
    test_npy_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy'),
    model_path: Annotated[Path, typer.Option] = Path('data/model.pth'),
    epochs: Annotated[int, typer.Option()] = 9, 
    batch_size: Annotated[int, typer.Option()] = 32, 
    lr: Annotated[float, typer.Option()] = .01
  ):

    print('---- Preprocessing ----')
    preprocess(
      train_in_path=train_csv_path,
      test_in_path=test_csv_path,
      train_out_path=train_npy_path,
      test_out_path=test_npy_path
    )

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    model = MNISTClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_dataloader, val_dataloader = get_dataloaders(train_npy_path, batch_size=batch_size)
    test_data = np.load(test_npy_path)
    X_test = torch.tensor(test_data[:, 1:]).float().reshape(test_data.shape[0], 1, 28, 28)
    y_test = torch.tensor(test_data[:, 0]).long()

    with start_tracker() as tracker:
      
      tracker.log_params({
        'loss_function': 'CrossEntropyLoss',
        'metric_function': 'Accuracy',
        'optimizer_class': 'SGD',
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
      })
      
      print('---- Training ----')
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

      print('---- Evaluation ----')
      eval(
        model=model, 
        X=X_test, 
        y=y_test, 
        loss_fn=loss_fn, 
        metric_fn=metric_fn, 
        tracker=tracker
      )


if __name__ == '__main__':
   typer.run(main)



 

  



