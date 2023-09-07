import torch
from torch.utils.data import DataLoader


def compute_mean_std(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates the mean and standard deviation of a dataset using a DataLoader.
    This function iterates through a DataLoader to compute the mean and
    standard deviation of the dataset based on the pixel values of the images.

    Args:
        dataloader (DataLoader): The DataLoader containing the dataset to compute
        statistics for.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two Tensors.
        The first Tensor represents the computed mean values for each
        channel (e.g., R, G, B), and the second Tensor represents
        the computed standard deviation values for each channel.
    """
    global_mean = 0.0
    global_std = 0.0

    for images, _ in dataloader:
        mean = 0.0
        std = 0.0

        for image in images:
            mean += image.mean((1, 2))
            std += image.std((1, 2))

        mean /= len(images)
        std /= len(images)

        global_mean += mean
        global_std += std

    global_mean /= len(dataloader)
    global_std /= len(dataloader)

    return global_mean, global_std
