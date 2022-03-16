import torch.nn as nn
import torch
import torch.nn.functional as F


class Deconv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Deconv3x3Block, self).__init__()
        self.add_module('deconv', nn.ConvTranspose2d(in_size, h_size, kernel_size=3, stride=2, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size))    

class Conv1x1Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv1x1Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=1, stride=1, padding=0, bias=True))

class Conv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv3x3Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size))    

class AvgBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(AvgBlock, self).__init__()
        self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    
        
class MaxBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(MaxBlock, self).__init__()
        self.add_module('pool', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    

class DownBlock(nn.Module):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, 
                 out_size: int, 
                 do_pool: int = True):
        super(DownBlock, self).__init__()     

        self.do_pool = do_pool
        self.pool = None
        if self.do_pool:
          self.pool = AvgBlock(kernel_size=2, stride=2, padding=0)

        in_size_cum = in_size  
        self.conv_1 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        self.conv_3 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        self.conv_2 = Conv1x1Block( in_size=in_size_cum,  h_size=out_size)

    def forward(self, x):
        
        batch_size = len(x)
        if self.do_pool:
          x = self.pool(x)
        x_list = []
        x_list.append(x)
        x = self.conv_1(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.conv_3(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        x = self.conv_2(x)
        return x

        

class UpBlock(nn.Module):
    def __init__(self, 
                 in_size:   int, 
                 in_size_2: int, 
                 h_size:    int, 
                 out_size:  int, 
                 ):
        super(UpBlock, self).__init__()     
        self.deconv   = Deconv3x3Block( in_size=in_size, h_size=h_size)
        self.out_conv = Conv3x3Block( in_size=h_size + in_size_2, h_size=out_size)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:4], scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)



class Unet(nn.Module):
    def __init__(self,):
        super(Unet, self).__init__()
        self.block0 = DownBlock(in_size=12, h_size=64, out_size=64, do_pool=False)
        self.block1 = DownBlock(in_size=64, h_size=96, out_size=96,)
        self.block2 = DownBlock(in_size=96, h_size=128, out_size=128, )
        self.block3 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block4 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block5 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block6 = DownBlock(in_size=128, h_size=128, out_size=128,)
        self.block20 = Conv3x3Block(in_size=128, h_size=128)
        self.block15 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block14 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,) 
        self.block13 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block12 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block11 = UpBlock(in_size=128, in_size_2=96 , h_size=128,  out_size=128,) 
        self.block10 = UpBlock(in_size=128, in_size_2=64 , h_size=128,  out_size=128,) 
        self.out_conv  = nn.Sequential(nn.Conv2d(128*1, 24, kernel_size=3, stride=1, padding=1, bias=True))
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                  nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                  nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = len(x)
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x  = self.block20(x6)
        x  = self.block15(x, x5)
        x  = self.block14(x, x4)
        x  = self.block13(x, x3)
        x  = self.block12(x, x2)
        x  = self.block11(x, x1)
        x  = self.block10(x, x0)
        x  = self.out_conv(x)
        x = x.view(batch_size, 24, 128, 128)
        return x