import argparse
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.modeling import build_model
from detectron2.utils import comm

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper2, MyDatasetMapper
from yolov7.utils.allreduce_norm import all_reduce_norm
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances


"""
This using for train instance segmentation!
"""

#os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg,
        #                                     mapper=MyDatasetMapper(cfg, True))
        cls.custom_mapper = MyDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=cls.custom_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model


def setup(args):
      # person dataset
    DATASET_ROOT = "./datasets/person/"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train2017")
    VAL_PATH = os.path.join(DATASET_ROOT, "val2017")
    TRAIN_JSON = os.path.join(ANN_ROOT, "person_train2017.json")
    VAL_JSON = os.path.join(ANN_ROOT, "person_val2017.json")
    register_coco_instances("person_train", {}, TRAIN_JSON, TRAIN_PATH) #mask_train
    register_coco_instances("person_val", {}, VAL_JSON, VAL_PATH) #mask_val
    
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    ## merge_from_list: can load pretrained weights?
    opts = ["MODEL.WEIGHTS", "output/myperson_yolomask/model_final.pth"]
    cfg.merge_from_list(opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
