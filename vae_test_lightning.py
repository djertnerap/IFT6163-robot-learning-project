import hydra

from rat_dataset import RatDataModule


def main():
    hydra.initialize(config_path="config", job_name="rat_random_walk_dataset_generator", version_base=None)
    config = hydra.compose(config_name="config_test")

    rat_data_module = RatDataModule(config["hardware"]["dataset_folder_path"])
    rat_data_module.prepare_data()
    rat_data_module.setup("train")
    data_loader = rat_data_module.train_dataloader()
    batch, _ = next(iter(data_loader))
    print(batch)
    print(batch.shape)


if __name__ == "__main__":
    main()
