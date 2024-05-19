from torch import nn
from stgcnplusplus.models import build_model
from mmcv.runner import load_checkpoint


class Stgcnplusplus_skeleton_freeze(nn.Module):

    def __init__(self, cfg, pretrained=None, std=0.01, dropout=0.0):
        super(Stgcnplusplus_skeleton_freeze, self).__init__()

        self.init_std = std
        self.dropout_ratio = dropout
        Stgcnplusplus = build_model(cfg)

        if pretrained is not None:
            load_checkpoint(Stgcnplusplus, pretrained)

        # for child in Stgcnplusplus.children():
        #     for param in child.parameters():
        #         param.requires_grad = False

        self.features = nn.Sequential(*list(Stgcnplusplus.children())[:-1])  # torch.Size([1, 1, 256, 25, 17])

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = cfg['cls_head']['in_channels']
        # self.num_classes = cfg['cls_head']['num_classes']
        # self.fc_cls = nn.Linear(self.in_c, self.num_classes)

    def forward(self, x):
        # bs, nc = x.shape[:2]
        # x = x.reshape((bs * nc, ) + x.shape[2:])
        # out = self.features(x)
    
        # 以下为torch.Size([1, 1, 256, 25, 17])的处理
        # if isinstance(out, list):
        #     for item in out:
        #         assert len(item.shape) == 2
        #     out = [item.mean(dim=0) for item in out]
        #     out = torch.stack(out)
        #
        # pool = nn.AdaptiveAvgPool2d(1)
        # N, M, C, T, V = out.shape
        # out = out.reshape(N * M, C, T, V)
        #
        # out = pool(out)
        # out = out.reshape(N, M, C)
        # out = out.mean(dim=1)
        # assert out.shape[1] == self.in_c
        #
        # if self.dropout is not None:
        #     out = self.dropout(out)

        # cls_score = self.fc_cls(out)
        # out = out.reshape(bs, nc, out.shape[-1])
        # assert len(out.shape) == 3  # * (Batch, NumSegs, Dim)
        # average_clips = self.test_cfg.get('average_clips', 'prob')
        # if average_clips not in ['score', 'prob', None]:
        #     raise ValueError(f'{average_clips} is not supported. Supported: ["score", "prob", None]')

        # if average_clips is None:
        #     return cls_score

        # if average_clips == 'prob':
        #     return F.softmax(out, dim=2).mean(dim=1)
        # elif average_clips == 'score':
        #     return cls_score.mean(dim=1)

        bs, nc = x.shape[:2]
        x = x.reshape((bs * nc, ) + x.shape[2:]) # bs*num_clips, numperson, flames, keypoints, xyscore
        out = self.features(x)
       
        # 以下为torch.Size([1, 1, 256, 25, 17])的处理        
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = out.shape  
        out = out.reshape(N * M, C, T, V)
        out = pool(out)
        out = out.reshape(N, M, C)
        out = out.mean(dim=1) # bs*numclips,256
        assert out.shape[1] == self.in_c
        out = out.reshape(bs, nc, out.shape[-1]) # bs,nc,256
        assert len(out.shape) == 3
        out = out.mean(dim=1) # bs,256
        if self.dropout is not None:
            out = self.dropout(out)

        return out 
