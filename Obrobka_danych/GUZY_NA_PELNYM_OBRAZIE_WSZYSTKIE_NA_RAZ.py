import pylidc as pl
from pylidc.utils import consensus
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.widgets import Slider

# Load the DICOM file and scan
print("Loading DICOM files ... This may take a moment.")
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == 'LIDC-IDRI-0020').first()
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

# Function to display a specific slice
def display_slice(slice_index, ax):
    # Get the slice of the volume and combined consensus mask
    slice_image = vol[:, :, slice_index]
    mask_slice = combined_cmask[:, :, slice_index]
    
    # Normalize the image to the range [0, 1]
    norm_slice = (slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image))
    
    # Create an RGB image from the normalized slice
    rgb_image = np.stack([norm_slice, norm_slice, norm_slice], axis=-1)
    
    # Apply red color to consensus mask pixels
    rgb_image[mask_slice, 0] = 1  # Red channel
    
    ax.clear()
    ax.imshow(rgb_image, cmap='gray')
    
    # Plot contours of the consensus mask
    for c in find_contours(mask_slice.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], '--r', linewidth=2, label='50% Consensus')
    
    ax.axis('off')
    plt.draw()

# Create a figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Initial display
display_slice(0, ax)

# Slider
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
num_slices = vol.shape[2]
slider = Slider(ax_slider, 'Slice', 0, num_slices-1, valinit=0, valstep=1)

def update(val):
    slice_index = int(val)
    display_slice(slice_index, ax)

slider.on_changed(update)

plt.show()
