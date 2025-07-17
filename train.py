import sys
import os
import torch
import torch.nn as nn
import wandb
# sys.path.append("/home2/saiteja3000/Dashcop_wsd/ultralytics/")
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


os.environ["WANDB_API_KEY"] = "3f76eaa43ff562e974332c3183e4cdfb4b4026f8"

wandb.init(project="divider_seg_yolov8x_july4annnots_nocurbwallside", job_type="training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

model = YOLO("yolov8x-seg.pt") 
print("Model task:", model.task)

add_wandb_callback(model, enable_model_checkpointing=False)

results = model.train(
    data="/home2/saiteja3000/Dashcop_wsd/divider_seg/data_experiments.yaml",
    epochs=100,
    batch=32,
    workers=20,
    device="0,1,2,3",
    project="/ssd_scratch/saiteja/projRideSafeDividerSeg",
    name="july4annots_yolov8x_nocurbwallside",
    task="segment",patience=20,
)

wandb.finish()
