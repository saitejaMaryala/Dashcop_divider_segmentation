# Dashcop Divider Segmentation

This repository is used to train a YOLO segmentation model for detecting different types of road dividers.

## 📄 File Descriptions

### `class_names.txt`
Contains the class names for different types of dividers.

### `data.yaml`
A YAML configuration file for training the YOLO model. It includes:
- Paths to the training and validation datasets
- The number of classes
- The list of class names

### `generate_labels_images.py`
Generates YOLO-compatible labels and corresponding images from raw data(xml files and videos) to prepare the training dataset.

### `generate_labels_images_parallel.py`
A parallelized version of the label and image generation script that utilizes multiple GPUs for faster processing.

### `train.py`
Script used to train the YOLO segmentation model using the provided dataset and configuration.

### `val.py`
Script used to validate the trained YOLO model on the validation dataset.

