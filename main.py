import hydra
import torch
from omegaconf import DictConfig

from vae import run_vae_experiment


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    if config["hardware"]["matmul_precision"]:
        torch.set_float32_matmul_precision(config["hardware"]["matmul_precision"])

    experiment_type = config["experiment_type"]
    if experiment_type == "vae":
        print(f"Running Experiment {experiment_type}...")
        run_vae_experiment(config)
    else:
        raise NotImplementedError(f"Experiment type {experiment_type} is unknown")


if __name__ == "__main__":
    main()
