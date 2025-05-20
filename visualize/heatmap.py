import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open_clip
from edition4paper.clip_ft_model import load_clip_model
import json
from tqdm import tqdm
import os

def get_patch_features_openclip(model, image_tensor):
    features = []

    def hook_fn(module, input, output):
        features.append(output)

    # 注册 hook：ViT 模型中，输出 patch 特征的是最后的 LayerNorm 之前那一层
    handle = model.visual.transformer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model.encode_image(image_tensor)

    handle.remove()

    # 返回 patch 特征 [1, num_tokens, dim]
    return features[0]

def load_image(image_path, preprocess, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image, image_tensor

# 3. 可视化 patch 特征作为热图（只看 patch 部分，不看 cls token）
def save_attention_heatmap(patch_features, original_image_path, save_path):
    patch_features = patch_features[:, 1:, :]  # 去掉 cls token, shape: (1, 257, 1024)
    patch_features = patch_features.squeeze(0).cpu().numpy()  # (256, 1024)

    # 平均池化得到每个 patch 的激活强度
    attention_map = patch_features.mean(axis=1)  # (256,)

    # 归一化到 [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    # reshape 成 14x14 格式（ViT-L-14 的 patch 分辨率）
    attention_map = attention_map.reshape(16, 16)

    # 放大到原图大小
    original_image = Image.open(original_image_path).convert("RGB")
    w, h = original_image.size
    attention_map_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 可选：平滑处理
    attention_map_resized = cv2.GaussianBlur(attention_map_resized, (11, 11), sigmaX=5)

    # 可视化叠加
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.35)
    plt.axis('off')

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def overlay_heatmap(original_image, heatmap, alpha=0.5):
    # heatmap 插值到图像大小
    heatmap_resized = cv2.resize(heatmap, original_image.size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    image_np = np.array(original_image)
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * image_np)

    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# json_path = r"D:\a_repo_pile\PPM-NET\utils\total5222_caption_dataset1.json"
json_path = r"D:\a_repo_pile\PPM-NET\utils\total620_caption_dataset2.json"
image_root = r"D:\a_repo_pile\stage2dataset\totalimages"
pretrained = r'D:\a_repo_pile\PPM-NET\edition4paper\weight_stratified_3adapter_100e_2loss_joint_dataset2stratified_dataset2.pth'
save_folder = r"D:\a_repo_pile\PPM-NET\visualize\heatmap_gaussian_hot"

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
model, preprocess = load_clip_model("ViT-L-14", pretrained, device)
model.to(device)
model.eval()

with open(json_path, "r") as f:
    test_data = json.load(f)
for item in tqdm(test_data, desc="Predicting"):
    file_name = item["image_filename"]  # +'.jpg'
    image_path = os.path.join(image_root, file_name)
    save_path = os.path.join(save_folder, f"{file_name.split('.')[0]}_heatmap.png")

    original_image, image_tensor = load_image(image_path, preprocess, device)

    patch_features = get_patch_features_openclip(model, image_tensor)
    save_attention_heatmap(patch_features, image_path, save_path)

print(f"Saved heatmap to {save_folder}")