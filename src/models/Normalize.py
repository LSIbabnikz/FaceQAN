
import torch
from torchvision.transforms import transforms

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).cuda()
        self.std = torch.tensor(std).cuda()

    def forward(self, input):
        x = input 
        x = transforms.Normalize(mean=self.mean, std=self.std)(x)
        return x