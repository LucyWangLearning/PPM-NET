import open_clip
import torch
import json
from tqdm import tqdm
import os
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from edition4paper.clip_ft_model import load_clip_model

# 降到二维空间
tsne = TSNE(n_components=2, random_state=42, perplexity=30)



json_path = r"D:\a_repo_pile\PPM-NET\utils\total620_caption_dataset2.json"
image_root = r"D:\a_repo_pile\stage2dataset\totalimages"
pretrained = r'D:\a_repo_pile\PPM-NET\edition4paper\weight_stratified_3adapter_100e_2loss_joint_dataset2stratified_dataset2.pth'
out_path_cls = r'tsne_image_cls_dataset2.jpg'
out_path_template = r'tsne_image_template_dataset2.jpg'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k", device=device)
# model, preprocess = load_clip_model("ViT-L-14", pretrained, device)
model.to(device)
model.eval()

features = []
template_labels = []
direction_labels = []

with open(json_path, "r") as f:
    test_data = json.load(f)
for item in tqdm(test_data, desc="Predicting"):
    file_name = item["image_filename"]  # +'.jpg'
    class_str = item["class"]  # 如 "left_3"
    class_template = item["class_template"]
    image_path = os.path.join(image_root, file_name)

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    features.append(image_features.cpu().numpy())
    direction_labels.append(class_str)
    template_labels.append(class_template)

import numpy as np
features = np.vstack(features)
features_tsne = tsne.fit_transform(features)
def plot_tsne(tsne_result, labels, title, output_path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='tab10', s=50)
    plt.title(title)
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


# Adapter 加入前
plot_tsne(features_tsne, template_labels, "Before Adapter - by Template", out_path_template)
plot_tsne(features_tsne, direction_labels, "Before Adapter - by Direction", out_path_cls)