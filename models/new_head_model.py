from torch import nn
import torch


class newmodel_freeze(nn.Module):

    def __init__(self, modelname, model, num_classes=None, drop_rate=0):
        super(newmodel_freeze, self).__init__()
        self.drop_rate = drop_rate 

        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
                # print(param.shape,param) 

        # ConvNeXt
        if modelname == 'ConvNeXt':
            self.model = model
            self.classifier = nn.Linear(768*6, num_classes)

        # GoogleNet
        if modelname == 'GoogleNet':
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.head = nn.Linear(1024*6, num_classes)

        # DenseNet
        if modelname == 'DenseNet':
            self.model = model
            self.classifier = nn.Linear(1024*6, num_classes)

        # EfficientNetV2
        if modelname == 'EfficientNetV2':
            self.model = model
            self.classifier = nn.Linear(1280*6, num_classes)

        # RegNet
        if modelname == 'RegNet':
            self.model = model
            self.classifier = nn.Linear(368*6, num_classes)
        
        # ResNeXt50_32x4d
        if modelname == 'ResNeXt50_32x4d':
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Linear(2048*6, num_classes)

        # SwinTransformer
        if modelname == 'SwinTransformer': 
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Linear(768*6, num_classes)

        # MobileViT
        if modelname == 'MobileViT':
            self.model = model
            self.classifier = nn.Linear(320*6, num_classes)

        # MobileNetV3
        if modelname == 'MobileNetV3':
            self.model = model
            self.classifier = nn.Linear(1024*6, num_classes)
        
        # ShuffleNet
        if modelname == 'ShuffleNet':
            self.model = model
            self.classifier = nn.Linear(1024*6, num_classes)

    def forward(self, x):
        x0 = self.model(x[0])
        if self.drop_rate > 0:
            x0 = nn.Dropout(self.drop_rate)(x0)
        x0 = x0.view(x0.size(0), -1)

        x1 = self.model(x[1])
        if self.drop_rate > 0:
            x1 = nn.Dropout(self.drop_rate)(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.model(x[2])
        if self.drop_rate > 0:
            x2 = nn.Dropout(self.drop_rate)(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.model(x[3])
        if self.drop_rate > 0:
            x3 = nn.Dropout(self.drop_rate)(x3)
        x3 = x3.view(x3.size(0), -1)

        x4 = self.model(x[4])
        if self.drop_rate > 0:
            x4 = nn.Dropout(self.drop_rate)(x4)
        x4 = x4.view(x4.size(0), -1)

        x5 = self.model(x[5])
        if self.drop_rate > 0:
            x5 = nn.Dropout(self.drop_rate)(x5)
        x5 = x5.view(x5.size(0), -1)

        x = torch.cat((x0,x1,x2,x3,x4,x5),dim=1)
        out = self.classifier(x)

        return out

