import os
import sys
# sys.path.append("/home2/saiteja3000/Dashcop_wsd/ultralytics/")
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

os.environ["WANDB_API_KEY"] = "3f76eaa43ff562e974332c3183e4cdfb4b4026f8"

wandb.init(project="divider_seg_yolov8x_july4annots_val")

model = YOLO("/ssd_scratch/cvit/saiteja/projRideSafeDividerSeg/july4annnots_yolov8x/weights/best.pt") 

model.val(
    data="/home2/saiteja3000/Dashcop_wsd/divider_seg/val_data.yaml",
    batch=32,
    device=[0],# Explicitly use all 4 GPUs
    project="/ssd_scratch/cvit/saiteja/divider_segmenation_results",  # base output folder
    name="val_results_july4annots_yolov8x_iou07_og",                              # subfolder
    iou = 0.7,
)

add_wandb_callback(model)

wandb.finish()


