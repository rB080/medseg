import segmentation_models_pytorch as smp
import torch.nn as nn

class smp_models(nn.Module):
    
    def __init__(self, model_key, nclass):
        super().__init__()
        model_dict = {
            "unet": smp.Unet,
            "upp": smp.UnetPlusPlus,
            "fpn": smp.FPN,
            "psp": smp.PSPNet,
            "man": smp.MAnet,
            "link": smp.Linknet
        }
        self.model = model_dict[model_key](encoder_name = 'resnet34',
                                           encoder_weights = 'imagenet',
                                           activation = 'sigmoid',
                                           in_channels = 3,
                                           classes=nclass)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        out = self.model(x)
        return out