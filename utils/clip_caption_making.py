import os
import json
import re  # 引入正则表达式模块
from PIL import Image

def generate_clip_training_data(image_20cls_mapping, image_16cls_mapping):
    # 根据字典{文件名：20类类别} 得到列表[字典{文件名，简短描述，长描述，类别描述，原始类别}]
    # 钟点方向映射
    result_list = []
    long_clock_map = {
        '3': "3 o'clock direction of the circle",
        '6': "6 o'clock direction of the circle",
        '9': "9 o'clock direction of the circle",
        '12': "12 o'clock direction of the circle",
        'one': "upper right quadrant of the circle",
        'two': "lower right quadrant of the circle",
        'three': "lower left quadrant of the circle",
        'four': "upper left quadrant of the circle",
        'nipple': "the exact center of the circle",
        'axilla': "the {direction} triangular area next to the circle"  # direction应该是图像级别的左右
    }
    short_clock_map = {
        '3': "directly to the right",
        '6': "directly below",
        '9': "directly to the left",
        '12': "directly above",
        'one': "at upper right quadrant",
        'two': "at lower right quadrant",
        'three': "at lower left quadrant",
        'four': "at upper left quadrant",
        'nipple': "at the exact center",
        'axilla': "at the {direction} triangular area"
    }
    # 从image_20cls_mapping获得20类的描述
    for image_name, direction_20_class  in image_20cls_mapping.items():

        direction, position_key = direction_20_class.split('_')

        if direction == 'right':  # 这里从医学上左右乳腺更改成图像级别的左右方位
            direction = "left"
        else:
            direction = "right"

        position_long_desc = long_clock_map.get(position_key, "")
        position_short_desc = short_clock_map.get(position_key, "")
        if position_key == 'axilla' and direction:
            position_long_desc = position_long_desc.format(direction=direction)  # 腋窝时需要区分左右。传入的是图像界别的direction
            position_short_desc = position_short_desc.format(direction=direction)  # 腋窝时需要区分左右。传入的是图像界别的direction
        caption_long = f"A photo of a template whose acute angle of the template points to the {direction}, with a short line segment is located at the {position_long_desc}."
        caption_short = f"A photo of a {direction}-pointing template, short line {position_short_desc}"

        # 从image_16cls_mapping获得类别模板
        base_filename = image_name.split('_crop')[0]+'.jpg'
        template_id = image_16cls_mapping.get(base_filename, None)
        if template_id is not None:
            caption_template = f"This image uses template type {template_id}."
        else:
            print(f"Template label not found for {base_filename}")
            caption_template = "Template type information not available."

        result_list.append({
            "image_filename": image_name+'.jpg',
            "caption_short": caption_short,
            "caption_long": caption_long,  # 新的模板编号 caption
            "caption_template": caption_template,
            "class": direction_20_class  # 这里的class_name是文件夹名字！也就是生理上的左右
        })

    return result_list

def save_as_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存至 {output_file}，共 {len(data)} 条记录")

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # 读取模板标签20映射，模板类别16映射
    image_20cls_json_path = r'D:\a_repo_pile\stage2dataset\dataset1\5222dataset1_template_label_mapping_20cls.json'
    image_16cls_json_path = r'D:\a_repo_pile\stage2dataset\dataset1\dataset1_template_class_mapping_16cls.json'
    output_file = r'total5222_caption_dataset1.json'
    image_20cls_mapping = open_json(image_20cls_json_path)
    image_16cls_mapping = open_json(image_16cls_json_path)
    # 生成训练数据
    data = generate_clip_training_data(image_20cls_mapping, image_16cls_mapping)
    save_as_json(data, output_file)
