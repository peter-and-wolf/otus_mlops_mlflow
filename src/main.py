from pathlib import Path

import typer
from typing_extensions import Annotated

import mlflow
import dvc.api

from scale import scale
from train import train
from eval import test


def main(
    train_path: Annotated[Path, typer.Option()] = Path('data/train_data.csv'),
    test_path: Annotated[Path, typer.Option()] = Path('data/test_data.csv'),
    model_path: Annotated[Path, typer.Option] = Path('data/model.pth'),
    loss_function: Annotated[str, typer.Option()] = 'CrossEntropyLoss',
    metric_function: Annotated[str, typer.Option()] = 'Accuracy',
    optimizer_class: Annotated[str, typer.Option()] = 'SGD',
    epochs: Annotated[int, typer.Option()] = 3, 
    batch_size: Annotated[int, typer.Option()] = 32, 
    lr: Annotated[float, typer.Option()] = .01
    ):
  
  train_scaled_path = train_path.with_suffix('.npy')
  test_scaled_path = test_path.with_suffix('.npy')

  mlflow.set_experiment('otus_1')

  with mlflow.start_run():

    mlflow.log_params({
      'train_data': dvc.api.get_url(train_path),
      'test_data': dvc.api.get_url(test_path), 
      'loss_function': loss_function,
      'metric_function': metric_function,
      'optimizer_class': optimizer_class,
      'batch_size': batch_size,
      'epochs': epochs,
      'lr': lr,
    })

    scale(
      train_in_path=train_path,
      test_in_path=test_path,
      train_out_path=train_scaled_path,
      test_out_path=test_scaled_path,
    )

    train(
      train_path=train_scaled_path,
      test_path=test_scaled_path,
      model_path=model_path,
      loss_function=loss_function,
      metric_function=metric_function,
      optimizer_class=optimizer_class,
      epochs=epochs,
      batch_size=batch_size,
      lr=lr
    )

    test(
      test_path=test_scaled_path,
      model_path=model_path,
      batch_size=batch_size,
      loss_function=loss_function,
      metric_function=metric_function
    )

if __name__ == '__main__':
  typer.run(main)
