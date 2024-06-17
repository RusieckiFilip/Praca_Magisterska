# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:38:28 2024

@author: Filip
"""

import pylidc as pl
from pylidc.utils import consensus
import numpy as np
import os

np.int = int
np.bool = bool

# Create the base directory for saving arrays
base_dir = 'Patients_tensors_numpy'
os.makedirs(base_dir, exist_ok=True)

# Query all patient scans
all_scans = pl.query(pl.Scan).all()

for scan in all_scans:
    patient_id = scan.patient_id
    
    # Create a directory for this patient if it doesn't already exist
    patient_dir = os.path.join(base_dir, patient_id)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir, exist_ok=True)
        
        # Load the volume for the scan
        vol = scan.to_volume()
        
        # Cluster annotations for the nodules in the scan
        nodules = scan.cluster_annotations()

        # Perform consensus on annotations
        clevel = 0.5  # 50% consensus level
        combined_cmask = np.zeros(vol.shape, dtype=bool)

        for nod in nodules:
            cmask, cbbox, masks = consensus(nod, clevel=clevel, pad=[(20, 20), (20, 20), (0, 0)])
            # Insert the consensus mask into the combined mask
            combined_cmask[cbbox] = np.logical_or(combined_cmask[cbbox], cmask)

        # Convert each slice of volume and mask to NumPy arrays
        slices = vol.shape[2]

        for i in range(slices):
            slice_image = vol[:, :, i]
            mask_slice = combined_cmask[:, :, i]
            
            # Save NumPy arrays in the patient's directory
            np.save(os.path.join(patient_dir, f'slice_{i}.npy'), slice_image)
            np.save(os.path.join(patient_dir, f'mask_{i}.npy'), mask_slice)
            
            # Print shapes for confirmation
            print(f"Patient {patient_id} - Slice {i} shape:", slice_image.shape)
            print(f"Patient {patient_id} - Mask {i} shape:", mask_slice.shape)
    else:
        print(f"Patient {patient_id} already processed.")

print("Processing complete for all patients.")
