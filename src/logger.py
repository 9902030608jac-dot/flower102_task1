from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        logger_type: str = "none",
        project: str = "flower102-task1",
        name: str | None = None,
        config: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
    ):
        self.logger_type = (logger_type or "none").lower()
        self.run = None
        self.backend = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.logger_type == "wandb":
            import wandb

            self.backend = wandb
            wandb_dir = str(self.log_dir) if self.log_dir is not None else None
            if wandb_dir is not None:
                os.environ.setdefault("WANDB_DIR", wandb_dir)
            self.run = wandb.init(project=project, name=name, config=config, dir=wandb_dir)
        elif self.logger_type == "swanlab":
            import swanlab

            self.backend = swanlab
            self.run = swanlab.init(project=project, experiment_name=name, config=config)
        elif self.logger_type == "none":
            self.backend = None
        else:
            raise ValueError(f"Unknown logger type: {logger_type}. Choose none, wandb, or swanlab.")

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.logger_type in {"wandb", "swanlab"}:
            self.backend.log(data, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        metric_name = name or "artifact"
        path_obj = Path(path)
        if self.logger_type == "wandb":
            if path_obj.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                self.backend.log({metric_name: self.backend.Image(str(path_obj))})
            return
        if self.logger_type == "swanlab":
            self.backend.log({metric_name: str(path_obj)})

    def save_file(self, path: str | Path, base_path: str | Path | None = None) -> None:
        path_obj = Path(path)
        if path_obj.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            return
        if self.logger_type == "swanlab":
            self.backend.log({"saved_file": str(path_obj)})
        # For WandB, keep files in outputs/ and only log scalar metrics/images.
        # Avoid wandb.save() so model weights and intermediate files are not uploaded.

    def finish(self) -> None:
        if self.run is not None and self.logger_type in {"wandb", "swanlab"}:
            self.backend.finish()
