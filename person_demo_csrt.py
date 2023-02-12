import argparse
import multiprocessing as mp
import pathlib
import random
import time
import os
import glob

import cv2
import detectron2.data.transforms as T
import torch
from alfred.utils.file_io import ImageSourceIter
from alfred.vis.image.det import visualize_det_cv2_part
from yolov7.utils.mask import vis_bitmasks_with_classes
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures.masks import BitMasks
from detectron2.utils.logger import setup_logger
from tqdm import trange
import numpy as np

from yolov7.config import add_yolo_config

# --------------------------- Tracker & Detector Comparing ----------------------------
# tracker's (x,y,w,h) = bbox
# everyone's (x,y,w,h) in rects
# compare these two information

# ----------------- IOU version -----------------

## box[0],[1]坐标为左上角(x,y)以及长与宽
def IOU_calculate(box1, box2):
    # box1 = [x1, y1, x2, y2]
    # box2 = [x, y, w, h]
    x1min, y1min = box1[0], box1[1]
    x1max, y1max = box1[2], box1[3]
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    s1 = w1 * h1

    x2min, y2min = box2[0], box2[1]
    x2max, y2max = box2[0] + box2[2], box2[1] + box2[3]
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0)
    inter_w = np.maximum(xmax - xmin, 0)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    IOU = intersection / union
    return IOU


# constants
WINDOW_NAME = "COCO detections"


class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            print("image after transform: ", image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            # predictions, pure_t = self.model([inputs])
            predictions = self.model([inputs])
            predictions = predictions[0]
            c = time.time() - tic
            print("cost: {}, fps: {}".format(c, 1 / c))
            return predictions


def setup_cfg(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.YOLO.CONF_THRESHOLD = args.confidence_threshold
    cfg.MODEL.YOLO.NMS_THRESHOLD = args.nms_threshold
    cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1
    # force devices based on user device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MAX_SIZE_TEST = 600  # 90ms
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/person/yolomask.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument(
        "-i",
        "--input",
        # nargs="+",
        # default="./images/mask/test6.jpg",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    #video自加line98
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "-o",
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.21,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "-n",
        "--nms-threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Name of Weights & Biases Project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Name of Weights & Biases Entity.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def vis_res_fast(res, img, class_names, colors, thresh, n):
    global tracker

    ins = res["instances"]
    bboxes = None
    if ins.has("pred_boxes"):
        # print("Yes")
        bboxes = ins.pred_boxes.tensor.cpu().numpy()
        # print(bboxes)
    
    # ---------------------------- change -------------------------------
    if n == 1:
        # simulate the process that selects a random person
        id = 0
        if len(bboxes) > 0: 
            id = np.random.randint(len(bboxes))
            box = bboxes[id]
        box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        box = [int(i) for i in box]
        # print(box)
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(img, box)
        print("Initialize!")
        track_id = id
    else:
        # Update tracker
        ok, box = tracker.update(img)
        # the object lost
        if (not ok):
            if len(bboxes) > 0: 
                id = np.random.randint(len(bboxes))
                box = bboxes[id]
            box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box = [int(i) for i in box]
            # Initialize tracker with first frame and bounding box
            ok = tracker.init(img, box)
            track_id = id
            print("Change!")
            print("Tracker bbox:", box)
        else:
            IOU_list = np.array([IOU_calculate(bbox, box) for bbox in bboxes])
            track_id = np.argmax(IOU_list)
    
    # track_id: id of objector we select
    # --------------------------------------------------------------------

    scores = ins.scores.cpu().numpy()
    clss = ins.pred_classes.cpu().numpy().astype(int)

    if ins.has("pred_bit_masks"):
        # print("Yes2")
        bit_masks = ins.pred_bit_masks

        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks, force_colors=colors, mask_border_color=(255, 255, 255), thickness=2)
        img = vis_bitmasks_with_classes(
            img, clss, bit_masks, force_colors=None, mask_border_color = (0, 255, 0), draw_contours=False, alpha=0.8, yolov_id = track_id
        ) #revise the masks
        
        '''if isinstance(bit_masks, torch.Tensor):
            bit_masks = bit_masks.cpu().numpy()
        assert isinstance(bit_masks, np.ndarray), 'bitmasks must be numpy array'
        bit_masks = bit_masks.astype(np.uint8)
        nn = 0
        for i, m in enumerate(bit_masks):
            nn += 1
            if m.shape != img.shape:
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
        
            cv2.imwrite(os.path.join('/home/disk1/liangxinyi/yolov7/bitmask_m', str(nn)+'bitmask_frame.jpg') ,m)'''
        
    if ins.has("pred_masks"):
        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        img = vis_bitmasks_with_classes(
            img,
            clss,
            bit_masks,
            force_colors=None,
            #mask_border_color = (0, 255, 0),
            draw_contours=True,
            alpha=0.6,
            thickness=2,
        )
        
    thickness = 1 if ins.has("pred_bit_masks") else 2
    font_scale = 0.3 if ins.has("pred_bit_masks") else 0.4
    if bboxes is not None:
        img = visualize_det_cv2_part(
            img,
            scores,
            clss,
            bboxes,
            class_names=class_names,
            force_color=colors,
            line_thickness=thickness,
            font_scale=font_scale,
            thresh=thresh,
        )
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = cfg.DATASETS.CLASS_NAMES
    predictor = DefaultPredictor(cfg)
    ##
    

    print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    colors = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in range(cfg.MODEL.YOLO.CLASSES)
    ]
    conf_thresh = cfg.MODEL.YOLO.CONF_THRESHOLD
    print("confidence thresh: ", conf_thresh)

    global tracker
    tracker = cv2.TrackerCSRT_create()

    if args.input:
        print("1")
        if os.path.isdir(args.input):
            imgs = glob.glob(os.path.join(args.input, '*.jpg'))
            imgs = sorted(imgs)
            for path in imgs:
                # use PIL, to be consistent with evaluation
                img = cv2.imread(path)
                print('ori img shape: ', img.shape)
                res = predictor(img)
                #res["instances"]
                res = vis_res_fast(res, img, metadata, colors)
                # test2
                '''blur = cv2.blur(img.bit_masks,(15,15),0)
                out = img.copy()
                mask = img.bit_masks
                out[mask>0] = blur[mask>0]'''

                # cv2.imshow('frame', res)
                cv2.imshow('frame', res)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        else:
            img = cv2.imread(args.input)
            res = predictor(img)
            #
            res = vis_res_fast(res, img, metadata, colors)
            # cv2.imshow('frame', res)
            cv2.imshow('frame', res)
            cv2.waitKey(0)
    elif args.webcam:
        print('Not supported.')
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        
        n = 1
        while(video.isOpened()):
            
            ret, frame = video.read()
            # frame = cv2.resize(frame, (640, 640))
            res = predictor(frame)
            #res = vis_res_fast(res, frame, metadata, colors, conf_thresh)
            res = vis_res_fast(res, frame, class_names, colors, conf_thresh, n)
            # cv2.imshow('frame', res)
            #cv2.imshow('frame', res)
            
            cv2.imwrite(os.path.join('video_results/demo_6', str(n)+'frame.jpg') ,res) #for jpg
            n += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    '''iter = ImageSourceIter(args.input)
    if args.wandb_project is not None:
        from wandadb.wandb_logger import WandbInferenceLogger

        inference_logger = WandbInferenceLogger(
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            conf_threshold=args.confidence_threshold,
            config=cfg,
        )
    else:
        inference_logger = None

    for i in trange(len(iter.srcs)):
        im = next(iter)
        if isinstance(im, str):
            image_path = im
            im = cv2.imread(im)
            res = predictor(im)
            if inference_logger:
                inference_logger.log_inference(image_path, res)

            res = vis_res_fast(res, im, class_names, colors, conf_thresh)
        # cv2.imshow('frame', res)
        if args.output:
            if pathlib.Path(args.output).is_dir():
                out_path = pathlib.Path(args.output) / pathlib.Path(image_path).name
            else:
                out_path = args.output
        else:
            out_path = "frame"
        cv2.imshow(out_path, res)
        # cv2.imwrite("test.jpg",res) #for jpg

        if iter.video_mode:
            cv2.waitKey(1)
        else:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                continue
    if inference_logger:
        inference_logger.finish_run()'''
