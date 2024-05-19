from torch import nn
import torch
from models.res_18 import res_18

class Res_onlypdface_freeze(nn.Module):

    def __init__(self, model=None, pretrained=None, checkpoint=None, num_classes=None, drop_rate=0):
        super(Res_onlypdface_freeze, self).__init__()
        self.drop_rate = drop_rate
        # 如果预训练模型是那个msceleb就用这个
        res_net_fer = model 
    
        if pretrained is not None:
            print("Loading pretrained weights...", pretrained)
            res_net_fer.load_state_dict(torch.load(pretrained))


        for child in res_net_fer.children():
            for param in child.parameters():
                param.requires_grad = False
                # print(param.shape,param)

        self.features = nn.Sequential(*list(res_net_fer.children())[:-2])
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x0 = self.features(x[0])
        if self.drop_rate > 0:
            x0 = nn.Dropout(self.drop_rate)(x0)
        x0 = x0.view(x0.size(0), -1)

        x1 = self.features(x[1])
        if self.drop_rate > 0:
            x1 = nn.Dropout(self.drop_rate)(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.features(x[2])
        if self.drop_rate > 0:
            x2 = nn.Dropout(self.drop_rate)(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.features(x[3])
        if self.drop_rate > 0:
            x3 = nn.Dropout(self.drop_rate)(x3)
        x3 = x3.view(x3.size(0), -1)

        x4 = self.features(x[4])
        if self.drop_rate > 0:
            x4 = nn.Dropout(self.drop_rate)(x4)
        x4 = x4.view(x4.size(0), -1)

        x5 = self.features(x[5])
        if self.drop_rate > 0:
            x5 = nn.Dropout(self.drop_rate)(x5)
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x0,x1,x2,x3,x4,x5),dim=1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out

