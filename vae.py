# VAE minimizing reconstruction error by distance squared |y_raw - y_hat_raw|^2

import hydra
from omegaconf import DictConfig
import torch


@hydra.main(config_path="config", config_name="config_vae")
def main(config: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['alg']['learning_rate'])
    
    for i in range(1, config['alg']['n_ter'] + 1):
        
        
        
        optimizer.zero_grad()
        
        model.
    pass


def train():
    pass


if __name__ == "__main__":
    main()

