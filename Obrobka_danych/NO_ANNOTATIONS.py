# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:52:54 2024

@author: Filip
"""

import pylidc as pl

# Query all scans
scans = pl.query(pl.Scan).all()

# Initialize counter for patients without annotations
patients_without_annotations = 0

# Check each scan for annotations
for scan in scans:
    # Get the patient ID
    patient_id = scan.patient_id
    
    # Get annotations for this scan
    annotations = scan.annotations
    
    # Check if there are no annotations
    if len(annotations) == 0:
        print(f"Patient {patient_id} has no annotations.")
        patients_without_annotations += 1

# Print the number of patients without annotations
print(f"Number of patients without annotations: {patients_without_annotations}")
