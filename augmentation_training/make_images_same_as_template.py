import ants
import antspynet

import os

import numpy as np



################################################
#
#  Load the data
#
################################################

print("Loading brain data.")


base_directory = '/media/share/AntsThings/test_batch/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template_file = base_directory + "maskedtemplate0-36um.nii.gz"

template = ants.image_read(template_file)

new_template_file = "/media/share/AntsThings/nick_visit/Volumes/Scan_0082__rec-36um.nrrd"
new_labels_file = "/media/share/AntsThings/nick_visit/Labels16-corrected/Scan_0082__rec-36um.nrrd"

new_template = ants.image_read(new_template_file)
new_labels = ants.image_read(new_labels_file)

reg = ants.registration(template, new_template, "Rigid")

oriented_new_template = reg['warpedmovout']
oriented_new_template_labels=ants.apply_transforms(template, new_labels, reg['fwdtransforms'], interpolator="genericLabel")

# check registration output
# ants.image_write(oriented_new_template, "/tmp/scan0082_rigid.nii.gz")
# ants.image_write(oriented_new_template_labels, "/tmp/scan0082_rigid-label.nii.gz")


new_spacing = (0.06, 0.06, 0.06)
new_shape = (192, 256, 192)

template2 = ants.resample_image(oriented_new_template, new_spacing, use_voxels=False, interp_type=4)
labels2 = ants.resample_image(oriented_new_template_labels, new_spacing, use_voxels=False, interp_type=1)

template2 = antspynet.pad_or_crop_image_to_size(template2, new_shape)
labels2 = antspynet.pad_or_crop_image_to_size(labels2, new_shape)

template2 = ants.iMath_normalize(template2)
#I don't think next line is necessary, since in this image there is no uniform 0 background.
template2[template2 == 0] = 0.1276
ants.image_write(template2, "/media/share/AntsThings/test_batch/template2-Scan_0082.nii.gz")
ants.image_write(labels2, "/media/share/AntsThings/test_batch/template2-labels-Scan0082.nii.gz")

