import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

verbose = True

base_directory = '/media/share/AntsThings/test_batch/'

#template_file = base_directory + "maskedtemplate0-36um.nii.gz"
#template_file = "/media/share/AntsThings/test_batch/maskedtemplate0-36um.nii.gz"
#template_file = "/media/share/AntsThings/nick_visit/Volumes/Scan_0082__rec-36um.nrrd"
template_file = "/tmp/scan00286_rigid.nii.gz"
template = ants.image_read(template_file)

new_spacing = (0.06, 0.06, 0.06)
new_shape = (192, 256, 192)

template = ants.resample_image(template, new_spacing, use_voxels=False, interp_type=4)
template = antspynet.pad_or_crop_image_to_size(template, new_shape)
template = ants.iMath_normalize(template)
template[template == 0] = 0.1276

image = ants.image_clone(template)

weights_file = "weights/initial.weights.h5"
# weights_file = "murat.weights.h5"

if verbose:
    print("Create model and load weights.")

number_of_filters = (16, 32, 64, 128, 256)
number_of_classification_labels = 17
unet_model = antspynet.create_unet_model_3d((*template.shape, 1),
   number_of_outputs=number_of_classification_labels, mode="classification", 
   number_of_filters=number_of_filters,
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2))
unet_model.load_weights(weights_file)

X = np.zeros((1, *template.shape, 1))
X[0,:,:,:,0] = image.numpy()

print("Prediction.")
Y = unet_model.predict(X, verbose=True)

probability_images = list()
for i in range(number_of_classification_labels):
    if verbose:
        print("Reconstructing image", i)
    prob_image = ants.from_numpy_like(np.squeeze(Y[0,:,:,:,i]), image)
    probability_images.append(prob_image)

image_matrix = ants.image_list_to_matrix(probability_images, image * 0 + 1)
segmentation_matrix = np.argmax(image_matrix, axis=0)
segmentation_image = ants.matrix_to_images(
    np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

ants.image_write(segmentation_image, "./predicted_segmentation.nii.gz")
