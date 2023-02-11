
# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
from pycocotools.coco import COCO, maskUtils
import json
import shutil
from tqdm import tqdm
 
classes_names = ['person']

 # 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
 
def main(args):
 
    coco = COCO(args.json_file)
 
    for cls in classes_names:
        catIds = coco.getCatIds(catNms=[cls])
        imgIds = coco.getImgIds(catIds=catIds)
 
        images_select_test = []
        annotations_select_test = []
 
        image_id_select_test = 0
        annotation_id_select_test = 0
 
 
        for imgId in tqdm(imgIds):
            img_info_append = []
            new_anns = []
            img_info = coco.loadImgs(imgId)[0]
 
            file_name = img_info['file_name']

            # split image
            img_path = 'yolov7/datasets/coco/train2017' + '/' + file_name   #***
            dst_img_dir = 'yolov7/datasets/person/train2017'
            mkr(dst_img_dir)
            dst_imgpath = dst_img_dir + '/' + file_name
            shutil.copy(img_path, dst_imgpath)
 
            # cvImage = cv2.imread(os.path.join(argv.input_file, file_name), -1)
            #
            # if cvImage is None:
            #     print('if cvImage is None:', file_name)
            #     exit()
 
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
 
            img_info_temp = img_info.copy()
            img_info_temp['id'] = image_id_select_test
            img_info_temp['file_name'] = file_name
            img_info_append.append(img_info_temp)
 
            for index in range(len(anns)):
                ann = anns[index]
 
                if 'segmentation' not in ann:  # 只处理存在annotation的情况
                    if type(ann['segmentation']) != list:
                        print("error no segmentation")
                        exit()
 
                ann_temp = ann.copy()
 
                ann_temp['id'] = annotation_id_select_test  # 对象ID
                ann_temp['image_id'] = image_id_select_test  # 图片ID
                new_anns.append(ann_temp)
 
                annotation_id_select_test += 1
 
            image_id_select_test += 1
            annotations_select_test.extend(new_anns)
            images_select_test.extend(img_info_append)
 
        instance_select_test2017 = {}
        instance_select_test2017['license'] = ['license']
        instance_select_test2017['info'] = 'spytensor created'
        
        instance_select_test2017['images'] = images_select_test
        instance_select_test2017['annotations'] = annotations_select_test
        instance_select_test2017['categories'] = [coco.dataset['categories'][catIds[0]-1]]
 
        import io
        output_path = args.output_file + '/' + 'person_train2017.json'
        # mkr(output_path)
        # os.path.join(args.output_file, "instances_train2017.json")
        with io.open(output_path, 'w', encoding="utf-8") as outfile:
            my_json_str = json.dumps(instance_select_test2017, ensure_ascii=False, indent=1)
            outfile.write(my_json_str)
 
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(
        description=
        "coco vis")
    parser.add_argument('-if',
                        "--input_file",
                        default='yolov7/datasets/coco/train2017/',
                        help="set input folder1")
    parser.add_argument('-oj',
                        "--json_file",
                        default='yolov7/datasets/coco/annotations/instances_train2017.json',
                        help="set input json")
    parser.add_argument('-of',
                        "--output_file",
                        default='yolov7/datasets/person/annotations',
                        help="set output folder")
    args = parser.parse_args()
 
    if args.output_file is None:
        parser.print_help()
        exit()
 
    main(args)
