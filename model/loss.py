import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.models import vgg19


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss"""

    def __init__(self, channels=1, **kwargs):
        """
        Initialize
        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        self.ssim_module = MS_SSIM(
            data_range=1023.0, size_average=True, win_size=3, channel=channels, **kwargs
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method
        Args:
            x: tensor one
            y: tensor two
        Returns: multi-scale SSIM Loss
        """
        return 1.0 - self.ssim_module(x, y)





class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg19 = vgg19(pretrained=True, progress=True).features[:-1].eval()
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.imagenet_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.loss = torch.nn.MSELoss(reduction="mean")
    
    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)        
        input = self.imagenet_normalize(input)
        target = self.imagenet_normalize(target)
        return self.loss(self.vgg19(input), self.vgg19(target))


