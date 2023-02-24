import torch
import torchvision.models as py_models
import torch.nn as nn

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        checkpoint = torch.load(model_path)
        vgg16 = py_models.vgg16(num_classes=2622)
        vgg16.load_state_dict(checkpoint['state_dict'])
        vgg16.eval()
        
        self.features1 = nn.Sequential(                
                *list(vgg16.features.children())[:3]
            )
        self.features2 = nn.Sequential(                
                *list(vgg16.features.children())[3:8]
            )
        self.features3 = nn.Sequential(                
                *list(vgg16.features.children())[8:15]
            )
        
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False
        for param in self.features3.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # C=64, W, H
        x1 = self.features1(x)
        # C=128, W/2, H/2
        x2 = self.features2(x1)
        # C=256, W/4, H/4
        x3 = self.features3(x2)
        return x1, x2, x3