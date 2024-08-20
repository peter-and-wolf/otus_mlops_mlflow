from typing import Any, Optional
from contextlib import contextmanager

import torch

import mlflow # type: ignore [import-untyped]
from mlflow.types import Schema, TensorSpec # type: ignore [import-untyped]
from mlflow.models import ModelSignature # type: ignore [import-untyped]


class MLFlowTracker:
  
  def start_run(self):
    mlflow.start_run()

  def stop_run(self):
    mlflow.end_run()

  def log_params(self, params: dict[str, Any]) -> None:
    mlflow.log_params(params) # type: ignore [attr-defined]

  def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
    mlflow.log_metric(name, value, step=step) # type: ignore [attr-defined]

  def log_model(self, 
                model: torch.nn.Module, 
                name: str, 
                code_paths: list[str] | None):
    mlflow.pytorch.log_model( # type: ignore [attr-defined]
      model, 
      name, 
      code_paths=code_paths
    )


@contextmanager
def start_tracker():
  tracker = MLFlowTracker()

  tracker.start_run()
  try:
    yield tracker
  finally:
    tracker.stop_run()



