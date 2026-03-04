from src.config import TrainConfig
from src.paths import ProjectPaths, get_project_root
from src.training import train_all


def main():
    cfg = TrainConfig()
    root = get_project_root(__file__)
    paths = ProjectPaths(root=root)
    train_all(paths, cfg)
    print("\n✅ Training complete. Check models/ and results/")


if __name__ == "__main__":
    main()
