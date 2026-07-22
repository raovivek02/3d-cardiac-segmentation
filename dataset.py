import os
import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    EnsureTyped
)

def get_acdc_data_dicts(data_dir):
    """
    Scans the database directory to create pairs of images and ground-truth masks.
    """
    patient_dirs = glob.glob(os.path.join(data_dir, "database", "training", "patient*"))
    data_dicts = []
    
    for p_dir in patient_dirs:
        # We target specific 3D frame states (e.g. frame01) for training
        # Match only the raw image files, avoiding ground truth files (*_gt.nii)
        images = sorted(glob.glob(os.path.join(p_dir, "*_frame[0-9][0-9].nii")))
        for img_path in images:
            gt_path = img_path.replace(".nii", "_gt.nii")
            if os.path.exists(gt_path):
                data_dicts.append({"image": img_path, "label": gt_path})
                
    return data_dicts

def get_transforms():
    """
    Defines the dictionary-based processing pipeline for 3D tensors.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),  # Min-max scale intensities to [0, 1]
       Resized(keys=["image", "label"], spatial_size=(256, 256, 16), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"])
    ])

if __name__ == "__main__":
    # Test our data pipeline logic locally
    data_dir = "./data"
    data_dicts = get_acdc_data_dicts(data_dir)
    print(f"Total 3D image-label pairs found: {len(data_dicts)}")
    
    transforms = get_transforms()
    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Load one single batch to verify tensor scaling and shape transformations
    first_batch = next(iter(dataloader))
    print("Processed Image Shape:", first_batch["image"].shape)
    print("Processed Label Shape:", first_batch["label"].shape)