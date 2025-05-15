import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import os
import json
import matplotlib.pyplot as plt  # ✅ 新增
from torch.cuda.amp import autocast
from clip_ft_model import load_clip_model, setup_trainable_params

from sklearn.metrics import accuracy_score
from PIL import Image

image_root = r'D:\a_repo_pile\stage2dataset\dataset1\5228images'
# 标签设定
direction_labels = ["Left-pointing template", "Right-pointing template"]
dir_label_map = {0: "left", 1: "right"}

position_labels_base = {
    '3': "3 o'clock direction of the circle",
    '6': "6 o'clock direction of the circle",
    '9': "9 o'clock direction of the circle",
    '12': "12 o'clock direction of the circle",
    'one': "upper right quadrant of the circle",
    'two': "lower right quadrant of the circle",
    'three': "lower left quadrant of the circle",
    'four': "upper left quadrant of the circle",
    'nipple': "the exact center of the circle",
    'axilla': "the triangular area next to the circle"
}
pos_keys = list(position_labels_base.keys())
pos_labels = [position_labels_base[k] for k in pos_keys]

def evaluate_accuracy(model, preprocess, val_data, dir_features, pos_features, device):
    pred_dir_labels = []
    pred_pos_labels = []
    gt_dir_labels = []
    gt_pos_labels = []

    model.eval()
    with torch.no_grad():
        for item in val_data:
            file_name = item["image_filename"]
            class_str = item["class"]
            dir_gt, pos_gt = class_str.split("_")
            if dir_gt == "left":
                dir_gt = "right"
            else:
                dir_gt = "left"
            image_path = os.path.join(image_root, file_name)

            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            dir_logits = image_features @ dir_features.T
            dir_pred = dir_logits.argmax(dim=1).item()
            pred_dir_labels.append(dir_pred)
            gt_dir_labels.append(0 if dir_gt == "left" else 1)

            pos_logits = image_features @ pos_features.T
            pos_pred = pos_logits.argmax(dim=1).item()
            pred_pos_labels.append(pos_pred)
            gt_pos_labels.append(pos_keys.index(pos_gt))

    dir_acc = accuracy_score(gt_dir_labels, pred_dir_labels)
    pos_acc = accuracy_score(gt_pos_labels, pred_pos_labels)

    return dir_acc, pos_acc


for exp_name in ["5shot", "10shot", "stratified"]:  #,"stratified""5shot", "10shot"
    print(f"\n========== 正在运行实验：{exp_name} ==========\n")

    # 路径变量
    train_json_path = fr'D:\a_repo_pile\PPM-NET\split_dataset\train_{exp_name}.json'
    val_json_path = fr'D:\a_repo_pile\PPM-NET\split_dataset\val_{exp_name}.json'
    keyword = f"{exp_name}_1(21)adapterMLPfus_100e"

    # -------------------- 配置 --------------------


    save_weights_path = f"weight_{keyword}.pth"
    best_weights_path = f"best_{save_weights_path}"  # ✅ 新增：保存最佳权重
    loss_plot_path = f"loss_curve_{keyword}.png"
    lambda_weight = 0.3
    num_epochs = 100
    batch_size = 32
    patience = 10  # ✅ 早停耐心轮数
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------- 加载数据 --------------------
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    train_data = load_json(train_json_path)
    val_data = load_json(val_json_path)

    # -------------------- 初始化模型 --------------------
    model, preprocess = load_clip_model('ViT-L-14', None, device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    model.to(device)
    model.eval()
    # ==== 编码文本 ====
    with torch.no_grad():
        dir_texts = tokenizer(direction_labels).to(device)
        dir_features = model.encode_text(dir_texts)
        dir_features /= dir_features.norm(dim=-1, keepdim=True)

        pos_texts = tokenizer(pos_labels).to(device)
        pos_features = model.encode_text(pos_texts)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)

    setup_trainable_params(model)
    model = model.to("cuda")

    # -------------------- 自定义 Dataset --------------------
    class CLIPDualCaptionDataset(Dataset):
        def __init__(self, data_list, image_root, preprocess, tokenizer):
            self.data_list = data_list
            self.image_root = image_root
            self.preprocess = preprocess
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            item = self.data_list[idx]
            image_path = os.path.join(self.image_root, item['image_filename'])
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image)

            text1 = self.tokenizer([item['caption_long']])[0]
            text2 = self.tokenizer([item['caption_template']])[0]

            return image, text1, text2


    # -------------------- 准备 Dataloader --------------------
    train_dataset = CLIPDualCaptionDataset(train_data, image_root, preprocess, tokenizer)
    val_dataset = CLIPDualCaptionDataset(val_data, image_root, preprocess, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------- 优化器和损失 --------------------
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=0.01  # 权重衰减项，防止过拟合
    )
    CLIPLoss = open_clip.loss.ClipLoss()

    # -------------------- 开始训练 --------------------
    train_losses, val_losses = [], []
    dir_accs, pos_accs = [], []

    best_pos_acc = 0.0
    best_epoch = 0
    no_improve_counter = 0

    log_file = open(f'loss_record_{keyword}.txt', 'w')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for images, texts1, texts2 in train_loader:
            images = images.to(device)
            texts1 = texts1.to(device)
            texts2 = texts2.to(device)

            optimizer.zero_grad()

            with autocast():
                image_features1, text_features1, logit_scale1 = model(images, texts1)
                image_features2, text_features2, logit_scale2 = model(images, texts2)

                loss_main = CLIPLoss(image_features1, text_features1, logit_scale=logit_scale1)
                loss_template = CLIPLoss(image_features2, text_features2, logit_scale=logit_scale2)

                total_loss = loss_main + lambda_weight * loss_template

            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, texts1, texts2 in val_loader:
                images = images.to(device)
                texts1 = texts1.to(device)
                texts2 = texts2.to(device)

                with autocast():
                    image_features1, text_features1, logit_scale1 = model(images, texts1)
                    image_features2, text_features2, logit_scale2 = model(images, texts2)

                    loss_main = CLIPLoss(image_features1, text_features1, logit_scale=logit_scale1)
                    loss_template = CLIPLoss(image_features2, text_features2, logit_scale=logit_scale2)
                    total_loss = loss_main + lambda_weight * loss_template

                total_val_loss += total_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # ✅ 每轮训练后计算 acc
        dir_acc, pos_acc = evaluate_accuracy(model, preprocess, val_data, dir_features, pos_features, device)
        dir_accs.append(dir_acc)
        pos_accs.append(pos_acc)

        log_msg = f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Dir Acc: {dir_acc:.4f}, Pos Acc: {pos_acc:.4f}"
        print(log_msg)
        log_file.write(log_msg + '\n')

        # ✅ 保存最佳模型（基于 pos_acc）
        if pos_acc > best_pos_acc:
            best_pos_acc = pos_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_weights_path)
            print(f"🔸 保存最佳模型 (Epoch {best_epoch}) -> {best_weights_path}")
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        # ✅ 提前停止
        if no_improve_counter >= patience:
            print(f"⏹️ 早停于第 {epoch + 1} 轮，最佳 Epoch 为 {best_epoch}")
            break

    log_file.close()

    # ✅ 保存最终模型
    torch.save(model.state_dict(), save_weights_path)
    print(f"✔️ 最终模型保存为 {save_weights_path}")

    plt.figure(figsize=(12, 6))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(dir_accs) + 1), dir_accs, label='Dir Accuracy')
    plt.plot(range(1, len(pos_accs) + 1), pos_accs, label='Pos Accuracy')
    plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"📈 损失和精度曲线已保存至 {loss_plot_path}")



