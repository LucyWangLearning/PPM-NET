import torch
import torch.nn as nn
from open_clip import create_model_and_transforms


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=768, residual=True):
        super().__init__()
        self.residual = residual
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)  # 不再压缩
        self.activation1 = nn.LeakyReLU(0.01)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation2 = nn.LeakyReLU(0.01)  # 新加的

    def forward(self, x):
        out = self.down_proj(x)
        out = self.activation1(out)
        out = self.up_proj(out)
        out = self.activation2(out)
        if self.residual:
            out = out + x
        return out


class AdapterBlockWrapper(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x, **kwargs):
        x = self.block(x, **kwargs)  # forward 原始 TransformerBlock
        x = self.adapter(x)  # 加 adapter（加残差）
        return x


def load_clip_model(model_name, pretrained, device):
    model, _, preprocess = create_model_and_transforms(model_name, pretrained='laion2b_s32b_b82k', device=device)

    # 插入 adapter 的层编号
    target_layers = [12, 18, 21]  # 0-based index
    embed_dim = model.visual.transformer.width  # ViT-L 的 embed dim 是 1024

    # 替换对应 transformer block
    for i in target_layers:
        original_block = model.visual.transformer.resblocks[i]
        adapter = Adapter(embed_dim, bottleneck_dim=768)
        model.visual.transformer.resblocks[i] = AdapterBlockWrapper(original_block, adapter)

    # # 先获取文本 transformer 的最后一层
    # text_block_idx = len(model.transformer.resblocks) - 1  # 最后一层的索引
    # text_block = model.transformer.resblocks[text_block_idx]
    # # 插入 Adapter（text embed_dim 通常也是 1024）
    # text_embed_dim = model.transformer.width  # 通常为 1024
    # text_adapter = Adapter(text_embed_dim, bottleneck_dim=768)
    # # 替换该层为 Adapter 包装的版本
    # model.transformer.resblocks[text_block_idx] = AdapterBlockWrapper(text_block, text_adapter)

    # 加载模型
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location=device))
    return model, preprocess
