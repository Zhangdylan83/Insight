import os
import nibabel as nib
import numpy as np
import argparse

def apply_lung_masks(nii_folder, mask_folder, output_folder):
    """
    Applies lung masks to NIfTI files and saves the lung-cropped images in FP32 format.
    Ensures that all pixels have the same precision and background intensity.

    Args:
    nii_folder (str): Directory containing the NIfTI files.
    mask_folder (str): Directory containing the corresponding mask files.
    output_folder (str): Directory where the lung-cropped NIfTI files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    nii_files = [f for f in os.listdir(nii_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]

    for nii_file in nii_files:
        base_name = nii_file.split('.')[0]
        mask_file = base_name + '.nii' 
        
        nii_path = os.path.join(nii_folder, nii_file)
        mask_path = os.path.join(mask_folder, mask_file)
        output_path = os.path.join(output_folder, base_name + '_crop.nii')

        if not os.path.exists(mask_path):
            print(f"Mask file not found for {nii_file}. Skipping.")
            continue

        
        image = nib.load(nii_path)
        mask = nib.load(mask_path)

        
        image_data = image.get_fdata().astype(np.float32)
        mask_data = mask.get_fdata().astype(np.int8)

        
        combined_mask = (mask_data == 1) | (mask_data == 2)  
        cropped_image_data = np.where(combined_mask, image_data, -1024).astype(np.float32)
        cropped_image = nib.Nifti1Image(cropped_image_data, affine=image.affine)

        
        nib.save(cropped_image, output_path)
        print(f"Cropped image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply lung masks to crop NIfTI files.")
    parser.add_argument(
        "--original_folder",
        type=str,
        required=True,
        help="Path to the folder containing original NIfTI files.",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        required=True,
        help="Path to the folder containing corresponding lung masks.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder where cropped NIfTI files will be saved.",
    )

    args = parser.parse_args()

    apply_lung_masks(args.original_folder, args.mask_folder, args.output_folder)
