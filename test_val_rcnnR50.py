#!/usr/bin/env python
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

import os
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

#prep MS COCO data into detectron2's standard format
from detectron2.data.datasets import register_coco_instances, load_coco_json

register_coco_instances("endeffector_train", {}, "endeffector_train/annotations/json_annotation_train.json", "endeffector_train/images")
register_coco_instances("endeffector_val", {}, "endeffector_val/annotations/json_annotation_val.json", "endeffector_val/images")

#run inference and create a predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.DATASETS.TRAIN = ("endeffector_train",)
cfg.DATASETS.TEST = ("endeffector_val", )
cfg.DATALOADER.NUM_WORKERS = 2
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

predictor = DefaultPredictor(cfg)

#store metadata
#MetadataCatalog.get("endeffector_val").set(thing_classes=["endeffector"])
end_metadata = MetadataCatalog.get("endeffector_val")

#visualize
from detectron2.utils.visualizer import ColorMode
dataset_dicts = load_coco_json("endeffector_val/annotations/json_annotation_val.json", "endeffector_val/images", "endeffector_val")
#dataset_dicts = register_coco_instances("endeffector_val", {}, "endeffector_val/annotations/json_annotation_val.json", "endeffector_val/images")
for d in dataset_dicts:    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=end_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('image', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#Evaluate performance using AP metric implemented in COCO API
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("endeffector_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "endeffector_val")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

