import json
import random
from collections import defaultdict
import os


def save_json(data, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{filename} 已保存，共 {len(data)} 条样本。")


def few_shot_split(data, k, seed):
    random.seed(seed)

    class_to_samples = defaultdict(list)
    for item in data:
        class_to_samples[item['class']].append(item)

    train_set, val_set, test_set = [], [], []

    for cls, samples in class_to_samples.items():
        samples = list(samples)
        random.shuffle(samples)

        if len(samples) < k:
            print(f"⚠️ 警告：类别 {cls} 样本数少于 k={k}，将跳过该类别")
            continue

        train_samples = samples[:k]
        remaining = samples[k:]
        val_size = max(1, int(len(remaining) * 0.2))

        train_set.extend(train_samples)
        val_set.extend(remaining[:val_size])
        test_set.extend(remaining[val_size:])

    return train_set, val_set, test_set


def stratified_split(data, k, seed):
    random.seed(seed)

    class_to_samples = defaultdict(list)
    for item in data:
        class_to_samples[item['class']].append(item)

    train_set, val_set, test_set = [], [], []

    for cls, samples in class_to_samples.items():
        samples = list(samples)
        random.shuffle(samples)
        total = len(samples)

        if total < 4:
            print(f"⚠️ 警告：类别 {cls} 样本数太少（{total} 个），可能无法满足比例划分。")

        n_train = int(total * k)
        n_val = int(total * (1 - k) / 2)
        n_test = total - n_train - n_val

        train_set.extend(samples[:n_train])
        val_set.extend(samples[n_train:n_train + n_val])
        test_set.extend(samples[n_train + n_val:])

    return train_set, val_set, test_set


def split_dataset(input_json, k, seed=42, output_dir="./"):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(k, int):
        train_set, val_set, test_set = few_shot_split(data, k, seed)
        prefix = f"{k}shot"
    elif isinstance(k, float) and 0 < k < 1:
        train_set, val_set, test_set = stratified_split(data, k, seed)
        prefix = f"stratified_{int(k)}"
    else:
        raise ValueError(
            "k should be either a positive integer (for k-shot) or a float between 0 and 1 (for stratified split)")

    save_json(train_set, f"train_{prefix}.json", output_dir)
    save_json(val_set, f"val_{prefix}.json", output_dir)
    save_json(test_set, f"test_{prefix}.json", output_dir)


if __name__ == "__main__":
    input_json_path = "total5222_caption_dataset1.json"
    output_dir = r"D:\a_repo_pile\PPM-NET\split_dataset"
    ks = [5, 10, 0.7]

    for k in ks:
        split_dataset(input_json=input_json_path, k=k, output_dir=output_dir)
