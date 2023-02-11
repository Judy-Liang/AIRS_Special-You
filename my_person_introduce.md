#############################
1.数据集分割代码
# 适用于coco格式数据集，提取person类别
my_split_person.py    # 不分图片，分json
split_person.py    # 分图片，分json

2.instance_seg数据集
"/home/disk1/liangxinyi/yolov7/datasets/person/"

3.config文件
"configs/person/yolomask.yaml"

4.训练程序
my_person_train_inseg.py（从train_inseg.py拷贝）
# 以下忽略
使用的annotation (train和val使用同一个coco2017val,比较小加快验证)
"/home/disk1/liangxinyi/yolov7/datasets/person2/annotations/person_test2017.json"

5.训练脚本
# COCO数据集
nohup python "/home/disk1/liangxinyi/yolov7/my_person_train_inseg.py"  --config-file "/home/disk1/liangxinyi/yolov7/configs/person/yolomask.yaml" --num-gpus 3 >my_person_train_inseg.log 2>&1 &
# 新数据集的训练（COCO格式）
nohup python "/home/disk1/liangxinyi/yolov7/my_person_train_inseg.py"  --config-file "/home/disk1/liangxinyi/yolov7/output/myperson_yolomask/config.yaml" --num-gpus 1 >my_person_train_inseg.log 2>&1 &

6.测试脚本(单张图片)
python "/home/disk1/liangxinyi/yolov7/my_demo.py" --config-file output/myperson_yolomask/config.yaml --input test_images/test.jpg --output test_result --opts MODEL.WEIGHTS output/myperson_yolomask/model_final.pth

7.demo数据集
"/home/disk1/liangxinyi/yolov7/videos/"
demo_1 - demo_9: 街景行人
ip1 & iPhone: 白墙背景自录
input: xxx.flv
output: "/home/disk1/liangxinyi/yolov7/video_results/"

8.person_demo介绍和指令
person_demo: 只有检测和分割
person_demo2: 检测分割+csrt跟踪
person_demo3: 检测分割+sort跟踪
指令：
python person_demo3.py --config-file output/myperson_yolomask/config.yaml  --video-input /home/disk1/liangxinyi/yolov7/videos/demo_6.flv -c 0.4 --opts MODEL.WEIGHTS output/person_yolomask/model_final.pth

9.person_demo_invisible介绍和指令
person_demo_invis: 锁定某一行人并跟踪，将初始无行人帧提取作为背景，将跟踪的行人轮廓合并到背景上，做到其他行人“消失”的视觉效果
指令：
python person_demo_invis.py --config-file output/myperson_yolomask/config.yaml --fg /home/disk1/liangxinyi/yolov7/fg_photo/iPhone_fg.jpg --video-input /home/disk1/liangxinyi/yolov7/videos/iPhone.flv -c 0.4 --opts MODEL.WEIGHTS output/person_yolomask/model_final.pth