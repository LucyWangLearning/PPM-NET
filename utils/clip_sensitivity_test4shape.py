import os
import random
import torch
import open_clip
from PIL import Image, ImageDraw
from tqdm import tqdm

# 参数
image_size = 224
num_images = 100
save_dir = "shape_test_images"
os.makedirs(save_dir, exist_ok=True)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# 定义 prompts
prompts = ["a photo of a circle", "a photo of a triangle"]
text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 图像生成函数
def draw_shape(shape_type):
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    if shape_type == "circle":
        r = random.randint(30, 50)
        x0 = random.randint(r, image_size - r)
        y0 = random.randint(r, image_size - r)
        draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill="blue")

    elif shape_type == "triangle":
        margin = 20
        p1 = (random.randint(margin, image_size - margin), random.randint(margin, image_size - margin))
        p2 = (random.randint(margin, image_size - margin), random.randint(margin, image_size - margin))
        p3 = (random.randint(margin, image_size - margin), random.randint(margin, image_size - margin))
        draw.polygon([p1, p2, p3], fill="blue")

    return img

# 推理并统计准确率
correct = 0
for i in tqdm(range(num_images)):
    true_label = random.choice(["circle", "triangle"])
    img = draw_shape(true_label)


    image_input = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T).softmax(dim=-1)

        pred = logits.argmax(dim=-1).item()
        prob = logits[0][pred]

    pred_label = "circle" if pred == 0 else "triangle"
    if pred_label == true_label:
        correct += 1
    img_path = os.path.join(save_dir, f"{i}_{true_label}_PRED{pred_label}_LOGITS{prob}.png")
    img.save(img_path)

print(f"\nTotal: {num_images} images")
print(f"Correct: {correct}")
print(f"Accuracy: {correct / num_images * 100:.2f}%")
