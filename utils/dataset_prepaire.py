# 又是码代码的一天呢：2025/5/13
# 数据集划分。传入原始images和labels路径，得到划分好的数据集
import os
import json
import random
import shutil
from collections import defaultdict

# 输入路径
images_dir = r"M:\dataset\stage1dataset\6426dataset1\images"
labels_dir = r"M:\dataset\stage1dataset\6426dataset1\labels"
json_file = r"M:\dataset\stage1dataset\6426dataset1\dataset1_template_class_mapping_16cls.json"

# 输出路径
output_dir = r"M:\dataset\stage1dataset\split_dataset"

# 创建所需的输出文件夹
output_train_dirs = ['1shot_train', '2shot_train', '5shot_train', 'val', 'test']
for folder in output_train_dirs:
    os.makedirs(os.path.join(output_dir, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, folder, 'labels'), exist_ok=True)

# 读取 JSON 映射文件
with open(json_file, 'r', encoding='utf-8') as f:
    file_to_class = json.load(f)

# 按类别分组图像
class_to_images = defaultdict(list)
for filename, class_id in file_to_class.items():
    class_to_images[class_id].append(filename)

# 创建抽样并分配数据集
def split_dataset():
    # 统计每个类的图片数量
    class_counts = {i: len(images) for i, images in class_to_images.items()}
    print("类的图片数量统计：", class_counts)

    # 用于保存每个类的数据
    train_1shot = defaultdict(list)
    train_2shot = defaultdict(list)
    train_5shot = defaultdict(list)
    val_images = defaultdict(list)
    test_images = defaultdict(list)

    for class_id, images in class_to_images.items():
        # 随机抽取5张，2张和1张图
        random.shuffle(images)
        if len(images) >= 5:
            train_5shot[class_id] = images[:5]
            remaining_images = images[5:]
        else:
            train_5shot[class_id] = images
            remaining_images = []

        if len(remaining_images) >= 2:
            train_2shot[class_id] = remaining_images[:2]
            remaining_images = remaining_images[2:]
        else:
            train_2shot[class_id] = remaining_images
            remaining_images = []

        if len(remaining_images) >= 1:
            train_1shot[class_id] = remaining_images[:1]
            remaining_images = remaining_images[1:]

        # 剩下的数据按2:8比例分为验证集和测试集
        split_index = int(len(remaining_images) * 0.2)
        val_images[class_id] = remaining_images[:split_index]
        test_images[class_id] = remaining_images[split_index:]

    return train_1shot, train_2shot, train_5shot, val_images, test_images


# 将抽样结果保存到目标文件夹
def save_to_folders(dataset, dataset_name):
    for class_id, images in dataset.items():
        for image in images:
            image_name = image
            label_name = image.replace('.jpg', '.txt')
            # 复制图像和标签
            shutil.copy(os.path.join(images_dir, image_name), os.path.join(output_dir, dataset_name, 'images', image_name))
            shutil.copy(os.path.join(labels_dir, label_name), os.path.join(output_dir, dataset_name, 'labels', label_name))

# 主函数
def main():
    train_1shot, train_2shot, train_5shot, val_images, test_images = split_dataset()

    # 保存到对应文件夹
    save_to_folders(train_1shot, '1shot_train')
    save_to_folders(train_2shot, '2shot_train')
    save_to_folders(train_5shot, '5shot_train')
    save_to_folders(val_images, 'val')
    save_to_folders(test_images, 'test')

    # 输出每个类的数据集数量
    for class_id in range(1, 17):  # 16个类
        print(f"类 {class_id}:")
        print(f"    1-shot 训练集数量: {len(train_1shot[class_id])}")
        print(f"    2-shot 训练集数量: {len(train_2shot[class_id])}")
        print(f"    5-shot 训练集数量: {len(train_5shot[class_id])}")
        print(f"    验证集数量: {len(val_images[class_id])}")
        print(f"    测试集数量: {len(test_images[class_id])}")

if __name__ == "__main__":
    main()
