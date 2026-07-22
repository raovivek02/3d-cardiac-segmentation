import os
import nibabel as nib
import numpy as np

def generate_augmented_samples():
    # 1. THE TRUTH: Inside database/ AND no .gz extension!
    source_file = "./data/database/testing/patient101/patient101_frame01.nii"
    output_dir = "./data/augumented_test_sample_images"
    
    print("Starting 3D data simulation pipeline...")

    # 2. Automatically create the missing folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Verified target directory exists: {output_dir}")
    
    # 3. Load the original raw test scan using an absolute path
    abs_source = os.path.abspath(source_file)
    try:
        img = nib.load(abs_source)
        data = img.get_fdata()
        print(f"Successfully loaded base scan: {abs_source}")
    except FileNotFoundError:
        print(f"Error: Could not find {abs_source}. Check the path!")
        return
    
    # 4. Simulate augmentation
    print("Applying synthetic 3D augmentation...")
    noise = np.random.normal(0, 0.02 * np.max(data), data.shape)
    augmented_data = data + noise
    
    # 5. Package and save the new augmented NIfTI file
    augmented_img = nib.Nifti1Image(augmented_data, img.affine, img.header)
    output_file = os.path.join(output_dir, "simulated_test_scan.nii.gz")
    nib.save(augmented_img, output_file)
    
    print(f"Success! Augmented 3D test sample saved to: {output_file}")

if __name__ == "__main__":
    generate_augmented_samples()