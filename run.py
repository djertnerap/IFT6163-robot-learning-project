import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config_test")
def main(config: DictConfig):
    for i in range(1, config['alg']['n_ter'] + 1):
#         We will probably need a replay buffer

        pass


if __name__ == "__main__":
    main()
