import os
from dataclasses import dataclass


def get_project_root(from_file: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(from_file), ".."))


@dataclass(frozen=True)
class ProjectPaths:
    root: str

    @property
    def data_dir(self) -> str:
        return os.path.join(self.root, "data")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.root, "models")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.root, "results")

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.data_dir, "spam.csv")

    def ensure_dirs(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
