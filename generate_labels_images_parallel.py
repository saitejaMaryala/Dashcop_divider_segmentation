import subprocess
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NUM_GPUS = 4
SCRIPT = "generate_labels_images.py"
CLASSES_TXT = "/home2/saiteja3000/Dashcop_wsd/divider_seg/class_names_experminent.txt"

VIDEO_ANNOT_PAIRS = []

# Define train/val paths
SETTINGS = [
    ("/ssd_scratch/cvit/keshav/dataset_wacv_v1/videoset3/original_videos", "/ssd_scratch/cvit/keshav/dataset_backup_cvat_30May25/RideSade_videoset3"),
    ("/ssd_scratch/cvit/keshav/dataset_wacv_v1/videoset2/original_videos", "/ssd_scratch/cvit/keshav/dataset_backup_cvat_30May25/RideSade_videoset2"),
    ("/ssd_scratch/cvit/keshav/dataset_wacv_v1/videoset1/original_videos", "/ssd_scratch/cvit/keshav/dataset_backup_cvat_30May25/RideSade_videoset1"),
    ("/ssd_scratch/cvit/keshav/dataset_wacv_v1/videoset5/original_videos", "/ssd_scratch/cvit/keshav/dataset_backup_cvat_30May25/RideSade_videoset5")
]

OUTPUT_IMAGES = {
    "videoset3": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/train/images",
    "videoset2": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/train/images",
    "videoset1": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/val/images",
    "videoset5": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/val/images",
}

OUTPUT_LABELS = {
    "videoset3": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/train/labels",
    "videoset2": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/train/labels",
    "videoset1": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/val/labels",
    "videoset5": "/ssd_scratch/cvit/saiteja/divider_segment_old_annots/val/labels",
}

# Collect all video/xml pairs
for video_dir, annot_dir in SETTINGS:
    for video_path in Path(video_dir).glob("*.mp4"):
        xml_path = Path(annot_dir) / f"{video_path.stem}.xml"
        if xml_path.exists():
            key = Path(video_dir).parent.name  # ✅ Fix here
            VIDEO_ANNOT_PAIRS.append((str(video_path), str(xml_path), OUTPUT_IMAGES[key], OUTPUT_LABELS[key]))
        else:
            print(f"⚠️ Missing XML for {video_path.name}")

def run_command(video, xml, out_img, out_lbl, gpu_id):
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        command = [
            "python3", SCRIPT,
            "--video", video,
            "--xml", xml,
            "--out_images", out_img,
            "--out_labels", out_lbl,
            "--classes", CLASSES_TXT
        ]
        print(f"[GPU {gpu_id}] Running: {' '.join(command)}")
        subprocess.run(command, check=True, env=env)
        return (video, True, None)
    except subprocess.CalledProcessError as e:
        return (video, False, str(e))

def run_parallel(pairs, num_gpus):
    successful, failed = [], []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(run_command, video, xml, out_img, out_lbl, i % num_gpus): (video, xml)
            for i, (video, xml, out_img, out_lbl) in enumerate(pairs)
        }

        for future in as_completed(futures):
            video, success, error = future.result()
            if success:
                print(f"✅ Success: {video}")
                successful.append(video)
            else:
                print(f"❌ Failed: {video}\n{error}")
                failed.append(video)

    print(f"\n--- Summary ---\nTotal: {len(pairs)} | Success: {len(successful)} | Fail: {len(failed)}")
    if failed:
        with open("failed_labels.txt", "w") as f:
            for v in failed:
                f.write(v + "\n")
        print("📝 Failed list saved to failed_labels.txt")

if __name__ == "__main__":
    run_parallel(VIDEO_ANNOT_PAIRS, NUM_GPUS)
