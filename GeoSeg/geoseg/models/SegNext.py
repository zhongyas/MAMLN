import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

class SegNext(nn.Module):
    def __init__(self, num_classes, in_channnels=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=config, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channnels, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
                               dropout=dropout, drop_path=drop_path)
        self.decoder = HamDecoder(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

def ft_unetformer(pretrained=True, num_classes=6, freeze_stages=-1, decoder_channels=256,
                  weight_path='pretrain_weights/stseg_base.pth'):
    model = SegNext(num_classes=num_classes,
                         freeze_stages=freeze_stages,
                         embed_dim=128,
                         depths=(2, 2, 18, 2),
                         num_heads=(4, 8, 16, 32),
                         decode_channels=decoder_channels)

    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model = SegNext()
    # #保存到txt文件
    # with open('ftunetformermodel.txt', 'w') as f:
    #     print(model, file=f)
    # #关闭文件
    # f.close()
    # print(model)
    x=torch.randn(4,3,512,512)
    y=model(x)
    print(y.shape)