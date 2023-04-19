import hydra
import torch
from omegaconf import DictConfig

from smp import run_smp_experiment
from vae import run_vae_experiment
from rl import run_rl_experiment


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    if config["hardware"]["matmul_precision"]:
        torch.set_float32_matmul_precision(config["hardware"]["matmul_precision"])

    experiment_type = config["experiment_type"]
    if experiment_type == "vae":
        print(f"Running Experiment {experiment_type}...")
        run_vae_experiment(config)
    elif experiment_type == "smp":
        run_smp_experiment(config)
    elif experiment_type == "rl":
        run_rl_experiment(config)
    else:
        raise NotImplementedError(f"Experiment type {experiment_type} is unknown")


if __name__ == "__main__":
    main()
