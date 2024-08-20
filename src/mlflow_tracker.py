from typing import Any, Optional
from contextlib import contextmanager

import torch

import mlflow # type: ignore [import-untyped]


class MLFlowTracker:
  
  def start_run(self):
    mlflow.start_run()

  def stop_run(self):
    mlflow.end_run()

  def log_params(self, params: dict[str, Any]) -> None:
    mlflow.log_params(params) # type: ignore [attr-defined]

  def log_metric(self, name: str, value: float, step: int | None = None) -> None:
    mlflow.log_metric(name, value, step=step) # type: ignore [attr-defined]

  def log_model(self,  
                name: str, 
                model: Any,
                input_example: Any | None,
                code_paths: list[str] | None):
    mlflow.pytorch.log_model( # type: ignore [attr-defined]
      model, 
      name, 
      input_example=input_example,
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



