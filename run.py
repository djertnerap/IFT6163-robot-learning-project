import hydra
from omegaconf import DictConfig

from spatial_memory_pipeline import SpatialMemoryPipeline
from utils.pytorch_util import init_gpu

import gymnasium as gym


@hydra.main(config_path="config", config_name="config_test")
def main(config: DictConfig):
    init_gpu(config['hardware']['use_gpu'], config['hardware']['which_gpu'])

    env = gym.make('MiniWorld-OpenField-v1', view="agent", render_mode="rgb_array")
    obs = env.reset(seed=config['logging']['seed'])

    agent = SpatialMemoryPipeline()
    for i in range(1, config['alg']['n_ter'] + 1):
#         We will probably need a replay buffer
#         Sample batch of episodes, then perform RNN training on all unroll lenghts of eac episode.

        pass


if __name__ == "__main__":
    main()
