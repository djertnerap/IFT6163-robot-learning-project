import os

import hydra
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from omegaconf import DictConfig

from rat_dataset import SequencedDataModule
from vae import LitAutoEncoder


def run_rl_experiment(config: DictConfig) -> None:
    """Run RL experiment."""
    original_cwd = hydra.utils.get_original_cwd()
    data_dir = os.path.abspath(original_cwd + config["hardware"]["smp_dataset_folder_path"])
    rat_sequence_data_module = SequencedDataModule(
        data_dir=data_dir,
        config=config,
        bptt_unroll_length=config["smp"]["bptt_unroll_length"],
        batch_size=config["smp"]["batch_size"],
        num_workers=config["hardware"]["num_data_loader_workers"],
        img_size=config["env"]["img_size"],
    )

    checkpoint_path = os.path.abspath(original_cwd + config["smp"]["ae_checkpoint_path"])
    ae = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path, in_channels=config["vae"]["in_channels"], net_config=config["vae"]["net_config"].values()
    )
    ae.eval()
    ae.freeze()

    # Need to handle the speed of the perpendicular direction of heading for RNN2
    smp = SpatialMemoryPipeline(
        batch_size=config["rlsmp"]["batch_size"],
        learning_rate=config["rlsmp"]["learning_rate"],
        memory_slot_learning_rate=config["rlsmp"]["memory_slot_learning_rate"],
        auto_encoder=ae,
        entropy_reactivation_target=config["rlsmp"]["entropy_reactivation_target"],
        memory_slot_size=config["vae"]["latent_dim"],
        nb_memory_slots=config["rlsmp"]["nb_memory_slots"],
        probability_correction=config["rlsmp"]["prob_correction"],
        probability_storage=config["rlsmp"]["prob_storage"],
        hidden_size_RNN1=config["rlsmp"]["hidden_size_RNN1"],
        hidden_size_RNN2=config["rlsmp"]["hidden_size_RNN2"],
        hidden_size_RNN3=config["rlsmp"]["hidden_size_RNN3"],
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        max_epochs=4, default_root_dir=original_cwd, logger=tb_logger, log_every_n_steps=1, profiler="simple"
    )
    trainer.fit(smp, datamodule=rat_sequence_data_module)

    # https://lightning-bolts.readthedocs.io/en/0.5.0/deprecated/models/reinforce_learn.html#soft-actor-critic-sac

    # # set seed
    # seed = config["hardware"]["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #
    # # set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # set up environment
    # env = gym.make(config["environment"]["name"])
    # env.seed(seed)
    # env = env.unwrapped
    # env = Monitor(env, directory=None, allow_early_resets=True)
    #
    # # set up agent
    # agent = Agent(
    #     state_size=env.observation_space.shape[0],
    #     action_size=env.action_space.n,
    #     seed=seed,
    #     config=config,
    # )
    #
    # # set up training
    # scores = train(agent, env, config)
    #
    # # save model
    # torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
    #
    # # plot scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel("Score")
    # plt.xlabel("Episode #")
    # plt.show()
    #
    # # close environment
    # env.close()
