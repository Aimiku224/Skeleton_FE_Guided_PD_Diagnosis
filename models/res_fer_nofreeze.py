from torch import nn
from models.res_18 import res_18

class Res_fer_nofreeze(nn.Module):

    def __init__(self, pretrained=None, checkpoint=None, num_classes=None, drop_rate=0):
        super(Res_fer_nofreeze, self).__init__()
        self.drop_rate = drop_rate

        res_net = res_18(pretrained=pretrained, num_classes=num_classes)

        self.features = nn.Sequential(*list(res_net.children())[:-1])
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out = self.fc2(x)

        return out

