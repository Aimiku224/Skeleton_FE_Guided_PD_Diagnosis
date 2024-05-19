from torch import nn
import torch
from models.res_18 import res_18

class Res_pd_freeze(nn.Module):

    def __init__(self, modelname, model=None, pretrained=None, checkpoint=None, num_classes=None, drop_rate=0):
        super(Res_pd_freeze, self).__init__()
        self.drop_rate = drop_rate
        res_net_fer = model
        # MobileNetV3
        if modelname == 'MobileNetV3':
            if pretrained is not None:
                weights_dict = torch.load(pretrained)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                    if 'classifier.3' not in k and model.state_dict()[k].numel() == v.numel()}
                print(model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found pretrained file: {}".format(pretrained))
            self.features = model
            
        if modelname == 'ResNet18':
            if pretrained is not None:
                print("Loading pretrained weights...", pretrained)
                res_net_fer.load_state_dict(torch.load(pretrained))
            self.features = nn.Sequential(*list(res_net_fer.children())[:-2])
        

    def forward(self, x):
        processed_x = []

        for i in range(len(x)):
            bs, num = x[i].shape[:2]
            x[i] = x[i].reshape((bs * num, ) + x[i].shape[2:])
            xi = self.features(x[i])
            
            if self.drop_rate > 0:
                xi = nn.Dropout(self.drop_rate)(xi)
            
            xi = xi.view(xi.size(0), -1)
            processed_x.append(xi)

        x0, x1, x2, x3, x4, x5 = processed_x

        out = torch.cat((x0,x1,x2,x3,x4,x5),dim=1)

        return out

