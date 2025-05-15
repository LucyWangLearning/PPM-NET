import os
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
import open_clip
from torchvision.transforms import ToPILImage

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L-14"
pretrained = "laion2b_s32b_b82k"
output_dir = "clip_location_test_images"
os.makedirs(output_dir, exist_ok=True)

# 加载模型和 tokenizer
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
tokenizer = open_clip.get_tokenizer(model_name)

# 图像生成参数
image_size = 224
shape_size = 50
positions = ["left", "right"] # "left", "right", "top", "bottom"
samples_per_position = 10  # 共 40 张图像

# 生成图像数据
def generate_image(position):
    img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if position == "left":
        x0 = random.randint(10, 30)
        y0 = random.randint(60, 160)
    elif position == "right":
        x0 = random.randint(image_size - shape_size - 30, image_size - shape_size - 10)
        y0 = random.randint(60, 160)
    elif position == "top":
        x0 = random.randint(60, 160)
        y0 = random.randint(10, 30)
    elif position == "bottom":
        x0 = random.randint(60, 160)
        y0 = random.randint(image_size - shape_size - 30, image_size - shape_size - 10)

    x1, y1 = x0 + shape_size, y0 + shape_size
    draw.ellipse([x0, y0, x1, y1], fill=(255, 0, 0))
    # draw.line([(0, img.height // 2), (img.width, img.height // 2)], fill=(0, 0, 0), width=1)  # 水平线
    draw.line([(img.width // 2, 0), (img.width // 2, img.height)], fill=(0, 0, 0), width=1)  # 垂直线
    return img

# 准备测试文本 prompt
prompts = {
    "left": "a photo of a shape in the left side of the image",
    "right": "a photo of a shape in the right side of the image",
    # "top": "a photo of a shape in the top part of the image",
    # "bottom": "a photo of a shape in the bottom part of the image"
}
text_tokens = tokenizer(list(prompts.values())).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 评估过程
correct = 0
total = 0

for pos in positions:
    for i in range(samples_per_position):
        img = generate_image(pos)

        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).softmax(dim=-1)
            pred_idx = similarity.argmax(dim=-1).item()
            pred_label = list(prompts.keys())[pred_idx]
            prob = similarity[0][pred_idx]

            is_correct = (pred_label == pos)
            correct += int(is_correct)
            total += 1
            print(f"[{pos}] → Predicted: {pred_label} | {'✓' if is_correct else '✗'}")
        img_path = os.path.join(output_dir, f"{pos}_{i}_PRED{pred_label}_LOGIT{prob}.png")
        img.save(img_path)

accuracy = correct / total * 100
print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
