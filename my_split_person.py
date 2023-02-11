import os
import cv2
import argparse
import numpy as np
from pycocotools.coco import COCO, maskUtils
import json
 
all_categories = [
    {
        "name": "person",
        "id": 1
    }
 ]
 
def main(argv):
 
    coco = COCO(argv.json_file)
 
    for m_key, m_val in enumerate(all_categories):
        print('m_key', m_key)
        print('m_val', m_val)
        catIds = coco.getCatIds(catNms=[m_val['name']])
        print('catIds',catIds)
        imgIds = coco.getImgIds(catIds=catIds)
        print('imgIds', len(imgIds))
 
        images_select_test = []
        annotations_select_test = []
 
        image_id_select_test = 0
        annotation_id_select_test = 0
 
 
        for i in range(len(imgIds)):
            img_info_append = []
            new_anns = []
            img_info = coco.loadImgs(imgIds[i])[0]
 
            file_name = img_info['file_name']
 
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
 
                if 'segmentation' not in ann:
                    if type(ann['segmentation']) != list:
                        print("error no segmentation")
                        exit()
 
                ann_temp = ann.copy()
 
                ann_temp['id'] = annotation_id_select_test 
                ann_temp['image_id'] = image_id_select_test 
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
        with io.open(os.path.join(args.output_file, "{}_val2017.json".format(m_val['name'])), 'w', encoding="utf-8") as outfile:
            my_json_str = json.dumps(instance_select_test2017, ensure_ascii=False, indent=1)
            outfile.write(my_json_str)
 
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(
        description=
        "coco vis")
    parser.add_argument('-if',
                        "--input_file",
                        default='./train2017/',
                        help="set input folder1")
    parser.add_argument('-oj',
                        "--json_file",
                        default='datasets/coco/annotations/instances_val2017.json',
                        help="set input json")
    parser.add_argument('-of',
                        "--output_file",
                        default='datasets/person2/annotations',
                        help="set output folder")
    args = parser.parse_args()
 
    if args.output_file is None:
        parser.print_help()
        exit()
 
    main(args)
