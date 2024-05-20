from pathlib import Path

import typer
from typing_extensions import Annotated

import numpy as np

import mlflow


def main(model_uri: Annotated[str, typer.Option()] = 'runs:/4c8a814ae0874942801781fe1de46c6a/model',
         data_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy'),
         index: Annotated[int, typer.Option(min=0, max=10_000)] = 42):
  
  # Load model as a PyFuncModel.
  loaded_model = mlflow.pyfunc.load_model(model_uri)
 
  data = np.load(data_path)
  X = data[:, 1:].reshape(data.shape[0], 1, 28, 28).astype(np.float32)
  y = data[:, 0]

  pred = loaded_model.predict(np.expand_dims(X[index], axis=0))

  print(f'predicted={np.argmax(pred)}, ground_truth={int(y[index])}')


if __name__ == '__main__':
  typer.run(main)