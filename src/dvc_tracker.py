from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

import torch

import dvclive


class DvcTracker:
  
  def __init__(self, live: dvclive.Live):
    self.live = live

  def log_params(self, params: dict[str, Any]) -> None:
    self.live.log_params(params) 

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    if step is not None:
      self.live.step = step
    self.live.log_metric(name, float(value))


  def log_model(self, 
                name: str, 
                model: Any, 
                input_example: Any | None,
                code_paths: list[str] | None):
    #model_path = f'{self.base_path}/{name}.pt'
    model_path = f'data/{name}.pt'
    torch.save(model.state_dict(), model_path)
    self.live.log_artifact(
      model_path,
      type='model',
      name=name,
    )


@contextmanager
def start_tracker():
  with dvclive.Live() as live:
    tracker = DvcTracker(live)
    yield tracker
