import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config_test")
def main(config: DictConfig):
    x = 1
    pass


if __name__ == "__main__":
    main()
