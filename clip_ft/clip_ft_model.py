import torch
import torch.nn as nn
from open_clip import create_model_and_transforms


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=768, output_dim=768, residual=True):
        super().__init__()
        self.residual = residual
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)  # 不再压缩
        self.activation1 = nn.LeakyReLU(0.01)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation2 = nn.LeakyReLU(0.01)

        self.final_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.down_proj(x)
        out = self.activation1(out)
        out = self.up_proj(out)
        out = self.activation2(out)
        if self.residual:
            out = out + x
        return self.final_proj(out)


class AdapterBlockWrapper(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x, **kwargs):
        x = self.block(x, **kwargs)  # forward 原始 TransformerBlock
        x = self.adapter(x)  # 加 adapter（加残差）
        return x
class CLIPWithAdapters(nn.Module):
    def __init__(self, clip_model, target_layers, adapter_bottleneck_dim=768):
        super().__init__()
        self.clip_model = clip_model
        self.target_layers = target_layers

        hidden_dim = clip_model.visual.transformer.width      # e.g., 1024
        output_dim = clip_model.visual.output_dim             # e.g., 768

        self.adapters = nn.ModuleList([
            Adapter(hidden_dim, adapter_bottleneck_dim, output_dim) for _ in target_layers
        ])

        self.raw_proj = nn.Identity()  # Already in output_dim

        # 融合权重（原始输出 + 每层adapter输出），可学习
        self.fusion_weights = nn.Parameter(torch.ones(len(target_layers) + 1))  # +1 for raw
        self.logit_scale = clip_model.logit_scale

    def encode_image(self, x):
        x = self.clip_model.visual._embeds(x)
        adapter_outputs = []

        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            x = block(x)
            if i in self.target_layers:
                idx = self.target_layers.index(i)
                adapter_out = self.adapters[idx](x)        # shape [B, N, H]
                adapter_outputs.append(adapter_out[:, 0])  # # [B, 768] 取 class token

        pooled, _ = self.clip_model.visual._pool(x)
        if self.clip_model.visual.proj is not None:
            pooled = pooled @ self.clip_model.visual.proj  # shape [B, D]


        # 将原始输出 + adapter输出堆叠: [N+1, B, D]
        features = [self.raw_proj(pooled)] + adapter_outputs
        stacked = torch.stack(features, dim=0)  # [N+1, B, D]

        # 融合权重归一化: [N+1, 1, 1]
        weights = torch.softmax(self.fusion_weights, dim=0)[:, None, None]
        fused = (weights * stacked).sum(dim=0)  # [B, D]

        return fused

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features, self.logit_scale.exp()


def load_clip_model(model_name, pretrained, device):
    clip_base, _, preprocess = create_model_and_transforms(model_name, pretrained='laion2b_s32b_b82k', device=device)
    model = CLIPWithAdapters(clip_base, target_layers=[12, 18, 21], adapter_bottleneck_dim=768).to(device)

    # 加载模型
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location=device))
    return model, preprocess

def setup_trainable_params(model):
    # 先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 只打开 adapter 和 logit_scale
    for name, param in model.named_parameters():
        if "adapter" in name or "fusion_weights" in name:
            param.requires_grad = True
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 输出信息
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable Ratio: {100 * trainable_params / total_params:.4f}%")

    print("\nTrainable modules:")
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                print(f"{name}.{param_name}: {param.numel()} params")