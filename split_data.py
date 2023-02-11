# 将图片和标注数据按比例切分为 训练集和测试集

import os
import random
from shutil import copy2

# 原始路径
image_original_path = "yolov7/datasets/person/images/val2017/"
label_original_path = "yolov7/datasets/person/annotations/val2017/"
# 上级目录
# parent_path = os.path.dirname(os.getcwd())
# parent_path = "D:\\AI_Find"
# 训练集路径
# train_image_path = os.path.join(parent_path, "image_data/seed/train/images/")
# train_label_path = os.path.join(parent_path, "image_data/seed/train/labels/")
train_image_path = os.path.join("yolov7/datasets/person/train/images/")
train_label_path = os.path.join("yolov7/datasets/person/train/annotations/")
# 测试集路径
test_image_path = os.path.join("yolov7/datasets/person/test/images/")
test_label_path = os.path.join("yolov7/datasets/person/test/annotations/")


# test_image_path = os.path.join(parent_path, 'image_data/seed/val/images/')
# test_label_path = os.path.join(parent_path, 'image_data/seed/val/labels/')


# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)


def main():
    mkdir()
    # 复制移动图片数据
    all_image = os.listdir(image_original_path)
    for i in range(len(all_image)):
        num = random.randint(1,5)#随机给图片赋值，每五个随机赋值一次，抽取不为2的图片
        if num != 2:
            copy2(os.path.join(image_original_path, all_image[i]), train_image_path)
            train_index.append(i)
        else:
            copy2(os.path.join(image_original_path, all_image[i]), test_image_path)
            val_index.append(i)

    # 复制移动标注数据
    all_label = os.listdir(label_original_path)
    for i in train_index:
            copy2(os.path.join(label_original_path, all_label[i]), train_label_path)
    for i in val_index:
            copy2(os.path.join(label_original_path, all_label[i]), test_label_path)


if __name__ == '__main__':
    train_index = []
    val_index = []
    main()