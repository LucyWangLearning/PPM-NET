import os
import random
import torch
import open_clip
from PIL import Image, ImageDraw
from tqdm import tqdm

# 参数设置
image_size = 224
num_images = 100
save_dir = "dot_relative_circle_test"
os.makedirs(save_dir, exist_ok=True)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# 文本 prompt 和类别映射
labels = ["above", "below", "left", "right"]
prompts = [f"a black dot {rel} a circle" for rel in labels]
text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 图像生成函数
def draw_dot_relative(label):  # label: above / below / left / right
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    # 画圆（蓝色）
    r = random.randint(30, 50)
    cx = random.randint(r + 30, image_size - r - 30)
    cy = random.randint(r + 30, image_size - r - 30)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="blue", width=3)

    # 黑点位置（加一点扰动）
    offset = r + 10
    jitter = random.randint(-5, 5)
    dot_radius = 4

    if label == "above":
        x = cx + jitter
        y = cy - offset
    elif label == "below":
        x = cx + jitter
        y = cy + offset
    elif label == "left":
        x = cx - offset
        y = cy + jitter
    elif label == "right":
        x = cx + offset
        y = cy + jitter

    draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill="black")
    return img

# 推理评估
correct = 0
for i in tqdm(range(num_images)):
    true_label = random.choice(labels)
    img = draw_dot_relative(true_label)


    image_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        pred = logits.argmax(dim=-1).item()

    pred_label = labels[pred]
    if pred_label == true_label:
        correct += 1
    img_path = os.path.join(save_dir, f"{i}_{true_label}_PRED{pred_label}.png")
    img.save(img_path)

print(f"\nTotal: {num_images} images")
print(f"Correct: {correct}")
print(f"Accuracy: {correct / num_images * 100:.2f}%")
