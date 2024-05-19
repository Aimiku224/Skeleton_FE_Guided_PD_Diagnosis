import torch
from torch import nn

class PD_skeandfer_freeze(nn.Module):

    def __init__(self, modelname, skemodel, pdfermodel, num_classes=None, flag=0):
        super(PD_skeandfer_freeze, self).__init__()
        stgcnplusplus = skemodel
        pdfer_res_net = pdfermodel
        self.flag = flag
        self.features1 = stgcnplusplus
        self.features2 = pdfer_res_net
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False
        # for name, param in self.features1.named_parameters():
        #    print(f'{name}: requires_grad={param.requires_grad}')
        if modelname == 'MobileNetV3':
            self.fc1 = nn.Linear(6144, 512)
            self.fc2 = nn.Linear(256, 512)
        else:
            self.fc1 = nn.Linear(3072, 512)
            self.fc2 = nn.Linear(256, 512)

        # Feature Fusion(Early Fusion)
        if self.flag==0:
            self.fc3 = nn.Linear(1024, num_classes)
        
        # Decision Fusion(Late Fusion)
        if self.flag==1:
            self.fc3 = nn.Linear(512, num_classes)
        
        # Hybrid Fusion
        if self.flag==2:
            self.out1 = nn.Linear(512, 1)
            self.out2 = nn.Linear(512, 1)
            self.fc3 = nn.Linear(513, num_classes)
        
        # Model Fusion(Incomplete...)
        if self.flag==3:
            self.fc2 = nn.Linear(512, num_classes)

    def forward(self, images, keypoints):
        
        # bs, nc = keypoints.shape[:2]
        # keypoints = keypoints.reshape((bs * nc, ) + keypoints.shape[2:])
        # pool = nn.AdaptiveAvgPool2d(1)
        # N, M, C, T, V = ske_features.shape # bs*num_clips, numperson, flames, keypoints, xyscore
        # ske_features = ske_features.reshape(N * M, C, T, V)
        # ske_features = pool(ske_features)
        # ske_features = ske_features.reshape(N, M, C)
        # ske_features = ske_features.mean(dim=1)  # bs*numclips,256
        # assert ske_features.shape[1] == self.in_c
        # ske_features = ske_features.reshape(bs, nc, ske_features.shape[-1]) # bs,nc,256
        # assert len(ske_features.shape) == 3
        # ske_features = ske_features.mean(dim=1) # bs,256
        # if self.dropout is not None:
        #     ske_features = self.dropout(ske_features)

        pdfer_features = self.features2(images)
        pdfer_mapping_features = self.fc1(pdfer_features)
        ske_features = self.features1(keypoints)
        ske_mapping_features = self.fc2(ske_features)

        if self.flag == 0:
            out = torch.cat((pdfer_mapping_features, ske_mapping_features), dim=1)
            out = self.fc3(out)

        if self.flag == 1:
            out = torch.mul(0.5, pdfer_mapping_features) + torch.mul(0.5, ske_mapping_features)
            out = self.fc3(out)

        if self.flag == 2:
            pdfer_out = self.out1(pdfer_mapping_features)
            pdfer_out = torch.cat((pdfer_out, pdfer_mapping_features), dim=1)
            ske_out = self.out2(ske_mapping_features)
            ske_out = torch.cat((ske_out, ske_mapping_features), dim=1)
            out = self.fc3(pdfer_out) + self.fc3(ske_out) 

        if self.flag == 3:
            out = self.fc2(out)

        return out
