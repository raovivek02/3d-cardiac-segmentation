import os
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from dataset import get_transforms 

def export_interactive_3d():
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting on device: {device}")

    # 2. Target the simulated test scan
    test_dir = "./data/augumented_test_sample_images/"
    test_file = os.path.join(test_dir, "simulated_test_scan.nii.gz")
    
    if not os.path.exists(test_file):
        print(f"Error: Could not find {test_file}. Check the simulation step!")
        return
        
    print(f"Processing raw test volume: {test_file}")

    # 3. Process data using training pipeline with a dummy label key
    transforms = get_transforms()
    data_dict = {"image": test_file, "label": test_file}
    
    processed_data = transforms(data_dict)
    input_tensor = processed_data["image"].unsqueeze(0).to(device)

    # 4. Model Definition
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # 5. Load Weights & Run Inference
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=device, weights_only=True))
    model.eval()

    print("Generating 3D segmentation mask...")
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=1)

    # 6. Extract the 3D Volume and save as NIfTI
    pred_volume = predictions[0].cpu().numpy().astype(np.float32)
    
    # Retain the exact affine metadata from the original simulation
    original_img = nib.load(test_file)
    nifti_img = nib.Nifti1Image(pred_volume, affine=original_img.affine)
    
    output_path = "ai_interactive_heart.nii.gz"
    nib.save(nifti_img, output_path)
    print(f"Success! Saved true 3D model to: {output_path}")

if __name__ == "__main__":
    export_interactive_3d()