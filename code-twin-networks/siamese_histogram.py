import torch
import numpy as np

from datasets import RandomOrganoidHistPairDataset
from model import SiameseNetwork, InputType
from resnet18_1d import resnet18_1d
from torch.utils.data import DataLoader

def params(model: torch.nn.Module):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def main():
    dataset = RandomOrganoidHistPairDataset(
        data_dirs=["/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/hist-dataset"],
        split="train"
    )

    loader = DataLoader(dataset, batch_size=16)

    for batch_idx, (hists1, hists2, targets) in enumerate(loader):
        print(batch_idx)



if __name__ == "__main__":
    main()