import os
import random
import torch
import open_clip
from PIL import Image, ImageDraw
from tqdm import tqdm

# 参数设置
image_size = 224
num_images = 100
save_dir = "dot_in_circle_test"
os.makedirs(save_dir, exist_ok=True)

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# 文本 prompt
prompts = ["a black dot inside a circle", "a black dot outside a circle"]
text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 生成图像函数
def draw_dot_in_or_out(label):  # label: "inside" or "outside"
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    # 画圆（蓝色）
    r = random.randint(40, 60)
    cx = random.randint(r + 10, image_size - r - 10)
    cy = random.randint(r + 10, image_size - r - 10)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline="blue", width=3)

    # 画黑点
    dot_radius = 3
    if label == "inside":
        angle = random.uniform(0, 2 * 3.1416)
        dist = random.uniform(0, r - 5)
        dx = int(dist * torch.cos(torch.tensor(angle)).item())
        dy = int(dist * torch.sin(torch.tensor(angle)).item())
        x = cx + dx
        y = cy + dy
    else:  # outside
        while True:
            x = random.randint(0, image_size - 1)
            y = random.randint(0, image_size - 1)
            if (x - cx) ** 2 + (y - cy) ** 2 > (r + 5) ** 2:
                break
    draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill="black")

    return img

# 推理并评估
correct = 0
for i in tqdm(range(num_images)):
    true_label = random.choice(["inside", "outside"])
    img = draw_dot_in_or_out(true_label)

    image_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T).softmax(dim=-1)

        pred = logits.argmax(dim=-1).item()
        prob = logits[0][pred]

    pred_label = "inside" if pred == 0 else "outside"
    if pred_label == true_label:
        correct += 1
    img_path = os.path.join(save_dir, f"{i}_{true_label}_PRED{pred_label}_LOGITS{prob}.png")
    img.save(img_path)

print(f"\nTotal: {num_images} images")
print(f"Correct: {correct}")
print(f"Accuracy: {correct / num_images * 100:.2f}%")
