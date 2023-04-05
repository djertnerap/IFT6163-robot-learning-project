import torch

# This code was inspired from the provided code in hte homeworks
device = None


def init_gpu(use_gpu: bool = True, gpu_id: int = 0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print(f"Using GPU id {gpu_id}")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def from_numpy(*args, **kwargs) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor: torch.Tensor):
    return tensor.to('cpu').detach().numpy()
