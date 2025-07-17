import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import argparse

def normalize_points(points, width, height):
    return [coord / dim for (coord, dim) in zip(sum(points, ()), [width, height] * len(points))]

def extract_annotated_frames_from_track_xml(video_path, xml_path, output_images, output_labels, class_names, class_counts):
    output_images = Path(output_images)
    output_labels = Path(output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    frame_annotations = {}

    for track in root.findall("track"):
        class_name = track.attrib.get("label")
        if class_name not in class_names:  # Skip 'curb' class
            continue
        class_id = class_names.index(class_name)

        for annotation in track:
            frame_idx = int(annotation.attrib.get("frame", -1))
            if frame_idx == -1:
                continue

            points = []
            if annotation.tag in ["polyline", "polygon"]:
                points_str = annotation.attrib.get("points", "")
                if points_str:
                    try:
                        points = [(float(x), float(y)) for x, y in (pair.split(",") for pair in points_str.split(";"))]
                    except ValueError:
                        continue  # skip invalid point formats
            elif annotation.tag == "box":
                try:
                    xtl = float(annotation.attrib["xtl"])
                    ytl = float(annotation.attrib["ytl"])
                    xbr = float(annotation.attrib["xbr"])
                    ybr = float(annotation.attrib["ybr"])
                    points = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)]
                except KeyError:
                    continue  # skip if any coordinates are missing

            if points:
                frame_annotations.setdefault(frame_idx, []).append((class_id, points))

    saved_count = 0
    for frame_idx, objs in frame_annotations.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue

        h, w = image.shape[:2]
        image_name = f"{video_path.stem}_frame_{frame_idx:05d}"
        image_path = output_images / f"{image_name}.jpg"
        label_path = output_labels / f"{image_name}.txt"

        label_lines = []
        for class_id, points in objs:
            class_counts[class_names[class_id]] += 1
            normalized_points = normalize_points(points, w, h)
            label_lines.append(f"{class_id} " + " ".join(f"{coord:.6f}" for coord in normalized_points))

        if label_lines:
            cv2.imwrite(str(image_path), image)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
            saved_count += 1

    cap.release()



if __name__ == "__main__":
    class_counts = Counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--out_images", type=str, required=True)
    parser.add_argument("--out_labels", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)

    args = parser.parse_args()

    class_names = Path(args.classes).read_text().splitlines()
    extract_annotated_frames_from_track_xml(Path(args.video), Path(args.xml), args.out_images, args.out_labels, class_names,class_counts)


# if __name__ == "__main__":
#     class_counts = Counter()
#     classes_txt = "/home2/saiteja3000/Dashcop_wsd/divider_seg/class_names_experminent.txt"   # path to the class names file
#     class_names = Path(classes_txt).read_text().splitlines()

#     # class_names = [
#     #     "curb",
#     #     "wall",
#     #     "sidewalk",
#     #     "single_solid_line",
#     #     "double_solid_line",
#     #     "police_barricade",
#     #     "cones",
#     #     "dashed_line",
#     #     "other_divider"
#     # ]

#     train_video_dirs = [
#         "/ssd_scratch/saiteja/dataset_wacv_v1/videoset3/original_videos",
#         "/ssd_scratch/saiteja/dataset_wacv_v1/videoset2/original_videos"
#     ]
#     train_annot_dirs = [
#         "/ssd_scratch/saiteja/RideSafe_annotations_4thJuly/RideSade_videoset3",
#         "/ssd_scratch/saiteja/RideSafe_annotations_4thJuly/RideSade_videoset2"
#     ]
#     train_output_images = "/ssd_scratch/saiteja/divider_segment_noCurbwallside/train/images"
#     train_output_labels = "/ssd_scratch/saiteja/divider_segment_noCurbwallside/train/labels"

#     val_video_dirs = [
#         "/ssd_scratch/saiteja/dataset_wacv_v1/videoset1/original_videos",
#        "/ssd_scratch/saiteja/dataset_wacv_v1/videoset5/original_videos"
#     ]
#     val_annot_dirs = [
#         "/ssd_scratch/saiteja/RideSafe_annotations_4thJuly/RideSade_videoset1",
#        "/ssd_scratch/saiteja/RideSafe_annotations_4thJuly/RideSade_videoset5"
#     ]
#     val_output_images = "/ssd_scratch/saiteja/divider_segment_noCurbwallside/val/images"
#     val_output_labels = "/ssd_scratch/saiteja/divider_segment_noCurbwallside/val/labels"

#     Process training videos
#     for video_dir, annot_dir in zip(train_video_dirs, train_annot_dirs):
#        for video_path in sorted(Path(video_dir).glob("*.mp4")):
#            xml_path = Path(annot_dir) / f"{video_path.stem}.xml"
#            if xml_path.exists():
#                extract_annotated_frames_from_track_xml(
#                    video_path, xml_path, train_output_images, train_output_labels, class_names, class_counts
#                )
#            else:
#                print(f"Missing XML for {video_path.name}")

#     # Process validation videos
#     for video_dir, annot_dir in zip(val_video_dirs, val_annot_dirs):
#         for video_path in sorted(Path(video_dir).glob("*.mp4")):
#             xml_path = Path(annot_dir) / f"{video_path.stem}.xml"
#             if xml_path.exists():
#                 extract_annotated_frames_from_track_xml(
#                     video_path, xml_path, val_output_images, val_output_labels, class_names, class_counts
#                 )
#             else:
#                 print(f"Missing XML for {video_path.name}")

#     # Print class counts
#     print("\n===== Annotation Summary =====")
#     for class_name in class_names:
#         print(f"{class_name}: {class_counts[class_name]} polygons")
