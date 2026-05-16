from __future__ import annotations

from typing import Any


class ExperimentLogger:
    def __init__(self, logger_type: str = "none", project: str = "flower102-task1", config: dict[str, Any] | None = None):
        self.logger_type = (logger_type or "none").lower()
        self.run = None
        if self.logger_type == "swanlab":
            import swanlab

            self.backend = swanlab
            self.run = swanlab.init(project=project, config=config)
        elif self.logger_type == "none":
            self.backend = None
        else:
            raise ValueError(f"Unknown logger type: {logger_type}. Wandb logging has been disabled for this project.")

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.logger_type == "swanlab":
            self.backend.log(data, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        metric_name = name or "artifact"
        if self.logger_type == "swanlab":
            self.backend.log({metric_name: str(path)})

    def save_file(self, path: str, base_path: str | None = None) -> None:
        if self.logger_type == "swanlab":
            self.backend.log({"saved_file": str(path)})

    def finish(self) -> None:
        if self.run is not None and self.logger_type == "swanlab":
            self.backend.finish()
