import json
from pathlib import Path
from typing_extensions import Annotated

import numpy as np
import requests

import typer


def main(model_url: Annotated[str, typer.Option()] = 'http://127.0.0.1:5002/invocations',
         data_path: Annotated[Path, typer.Option()] = Path('data/test_data.npy'),
         index: Annotated[int, typer.Option(min=0, max=10_000)] = 42):
  
  data = np.load(data_path)
  X = data[:, 1:].reshape(data.shape[0], 1, 28, 28).astype(np.float32)

  response = requests.post(
    url=model_url,
    data=json.dumps({
      'instances': np.expand_dims(X[index], axis=0).tolist()
    }),
    headers={
      'Content-Type': 'application/json'
    },
  )

  print(f'my prediction is {np.argmax(response.json()['predictions'])}')


if __name__ == '__main__':
  typer.run(main)