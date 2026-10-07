import fiftyone as fo
import fiftyone.zoo as foz

# Load the dataset (if not already loaded)
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person"],
    max_samples=2000  # optional, reduce for faster export
)

# Export in YOLOv5 format
dataset.export(
    export_dir="C:/Users/ndivi/SafetyAlertProject/data/person_coco",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    classes=["person"]
)

print("Export complete!")
