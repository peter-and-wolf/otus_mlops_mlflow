from typing import Any, Protocol

import torch


class ExperimentTracker(Protocol):
    
  def log_params(self, params: dict[str, Any]) -> None:
    """ Logs a bunch of hyperparams """

  def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
    """ Logs a single metric with name and value """

  def log_model(self, 
                model: torch.nn.Module, 
                name: str, 
                input_example: Any | None,
                code_paths: list[str] | None):
    """ Logs a model as an artefact """


