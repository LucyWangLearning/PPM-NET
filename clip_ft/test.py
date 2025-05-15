
import json
from open_clip import tokenize
from clip_ft_model import load_clip_model
import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

model_name = 'ViT-L-14'

def draw_confusion_matrix(gt_labels, pred_labels, keys, confusion_mat_pos_path):
    # 绘制混淆矩阵
    cm = confusion_matrix(gt_labels, pred_labels)
    # 类别名称
    class_names = keys
    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(confusion_mat_pos_path, dpi=300)  # 保存图像
    plt.show()

for exp_name in ['train', "val", 'test']: #, "stratified"   "5shot", "10shot",
    print(f"\n========== 正在运行实验：{exp_name} ==========\n")
    import sys
    # 指定输出文件路径
    log_file = f"eval_output_{exp_name}_10shot.txt"
    confusion_mat_pos_path = f"confusion_matrix_pos_{exp_name}_10shot.png"
    confusion_mat_dir_path = f"confusion_matrix_dir_{exp_name}_10shot.png"
    # 保存旧的 stdout
    original_stdout = sys.stdout
    # ========== 配置 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_root = r'D:\a_repo_pile\stage2dataset\dataset1\5228images'
    # image_root = r'D:\datasetTotal\CLIPstage2\dataset2\single_815-19classified'
    json_path = fr'D:\a_repo_pile\PPM-NET\split_dataset\{exp_name}_10shot.json'
    # json_path = r'D:\PROGRAM\lite_ICON\pythonProject1\multitask_clip\split_dataset\dataset2_total.json'
    # pretrained = fr"D:\PROGRAM\lite_ICON\pythonProject1\multitask_clip\best_weight_loss3_e50_{exp_name}.pth"
    pretrained = fr"D:\a_repo_pile\PPM-NET\clip_ft\best_weight_10shot_3adapter_100e.pth"

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

    # ==== 加载模型 ====
    model, preprocess = load_clip_model(model_name, pretrained, device)
    model.to(device)
    model.eval()
    # ==== 编码文本 ====
    with torch.no_grad():
        dir_texts = tokenize(direction_labels).to(device)
        dir_features = model.encode_text(dir_texts)
        dir_features /= dir_features.norm(dim=-1, keepdim=True)

        pos_texts = tokenize(pos_labels).to(device)
        pos_features = model.encode_text(pos_texts)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)

    # ==== 单张图预测函数 ====
    def predict_single(image_path):
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 方向预测
            dir_logits = image_features @ dir_features.T
            dir_pred_idx = dir_logits.argmax(dim=1).item()
            dir_pred = dir_label_map[dir_pred_idx]

            # 位置预测
            pos_logits = image_features @ pos_features.T
            pos_pred_idx = pos_logits.argmax(dim=1).item()
            pos_pred = pos_keys[pos_pred_idx]

            # 获取 top3 位置
            pos_top3_indices = pos_logits.topk(3, dim=1).indices.squeeze().tolist()
            pos_top3 = [pos_keys[i] for i in pos_top3_indices]

        return dir_pred, pos_pred, pos_top3, pos_logits.squeeze().cpu().numpy()


    # ==== 多图预测与评估 ====
    def evaluate(dataset_dir,json_path):
        gt_dir_labels = []
        gt_pos_labels = []
        pred_dir_labels = []
        pred_pos_labels = []
        pos_logits_all = []
        gt_pos_idx_all = []

        with open(json_path, "r") as f:
            test_data = json.load(f)

        # 遍历每张图进行预测
        for item in tqdm(test_data, desc="Predicting"):
            file_name = item["image_filename"] #+'.jpg'
            class_str = item["class"]  # 如 "left_3"
            image_path = os.path.join(dataset_dir, file_name)

            # 解析 ground truth
            dir_gt, pos_gt = class_str.split("_")
            if dir_gt == "left":
                dir_gt = "right"
            else:
                dir_gt = "left"

            # 模型预测
            dir_pred, pos_pred, pos_top3, pos_logits = predict_single(image_path)

            # 保存标签和预测
            gt_dir_labels.append(dir_gt)
            gt_pos_labels.append(pos_gt)
            pred_dir_labels.append(dir_pred)
            pred_pos_labels.append(pos_pred)

            # 为 map 准备数据
            pos_logits_all.append(pos_logits)
            gt_pos_idx_all.append(pos_keys.index(pos_gt))
        # 打开文件并将 stdout 重定向到文件
        with open(log_file, 'w', encoding='utf-8') as f:
            sys.stdout = f  # 重定向
            # ==== Direction 评估 ====
            print("\n=== Direction Evaluation ===")
            print(classification_report(gt_dir_labels, pred_dir_labels, digits=4))

            # ==== Position 评估 ====
            print("\n=== Position Top-1 Accuracy ===")
            print("Accuracy:", accuracy_score(gt_pos_labels, pred_pos_labels))

            print("\n=== Position P/R/F1 ===")
            print(classification_report(gt_pos_labels, pred_pos_labels, digits=4))

            # ==== Position Top-3 Accuracy ====
            print("\n=== Position Top-3 Accuracy ===")
            top3_correct = 0
            for i, logits in enumerate(pos_logits_all):
                top3_preds = np.argsort(logits)[-3:][::-1]  # 从大到小排序取前3
                if gt_pos_idx_all[i] in top3_preds:
                    top3_correct += 1
            top3_acc = top3_correct / len(gt_pos_idx_all)
            print(f"Top-3 Accuracy: {top3_acc:.4f}")

            # ==== mAP（mean Average Precision）====
            print("\n=== Position mAP ===")
            from sklearn.metrics import average_precision_score

            # 构造 one-hot 形式的 ground truth（N 张图，K 个类别）
            num_samples = len(pos_logits_all)
            num_classes = len(pos_keys)
            y_true = np.zeros((num_samples, num_classes))
            y_scores = np.vstack(pos_logits_all)  # shape: [N, K]

            for i, gt_idx in enumerate(gt_pos_idx_all):
                y_true[i, gt_idx] = 1.0

            # 计算每个类别的 average precision
            APs = []
            for class_idx in range(num_classes):
                ap = average_precision_score(y_true[:, class_idx], y_scores[:, class_idx])
                if not np.isnan(ap):
                    APs.append(ap)

            mean_ap = np.mean(APs)
            print(f"mean Average Precision (mAP): {mean_ap:.4f}")
        sys.stdout = original_stdout
        print(f"评估结果已保存到：{log_file}")

        draw_confusion_matrix(gt_pos_labels, pred_pos_labels, pos_keys, confusion_mat_pos_path)

    # ==== 执行评估 ====
    evaluate(image_root, json_path)  # 替换为你的图片文件夹路径
