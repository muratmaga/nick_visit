import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

verbose = True

# template_url = "https://figshare.com/ndownloader/files/52744724"
# weights_url = "https://figshare.com/ndownloader/files/52744397"
# template_file = tf.keras.utils.get_file("ferretT1Template.nii.gz", template_url)
# weights_file = tf.keras.utils.get_file("ferretT1wBrainExtraction3D.weights.h5", weights_url)

base_directory = '/media/share/AntsThings/test_batch/'
template_file = base_directory + "image.nii.gz"
template = ants.image_read(template_file)

# Just clone the template to simplify the example
# image = ants.image_clone(template)
# image = ants.add_noise_to_image(image, noise_model="additivegaussian", noise_parameters = (0.0, 0.01))
# for real images comment out the two lines above, and just read it as image 


new_spacing = (0.06, 0.06, 0.06)
new_shape = (192, 256, 192)

template = ants.resample_image(template, new_spacing, use_voxels=False, interp_type=4)
template = antspynet.pad_or_crop_image_to_size(template, new_shape)
template = ants.iMath_normalize(template)
template[template == 0] = 0.1276

image = ants.image_clone(template)

weights_file = "weights/initial.weights.h5"

print("Warping to template.")
# replace all of this with affine transform of the target to the template. 

#center_of_mass_reference = ants.get_center_of_mass(template * 0 + 1)
#center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
#translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
#xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
#    center=np.asarray(center_of_mass_reference), translation=translation)
#xfrm_inv = ants.invert_ants_transform(xfrm)

#image_warped = ants.apply_ants_transform_to_image(xfrm, image, template, interpolation="linear")
#image_warped = ants.iMath_normalize(image_warped)
image_warped = ants.image_clone(image)
ants.image_write(image_warped, "/tmp/image_warped.nii.gz")

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
X[0,:,:,:,0] = image_warped.numpy()

print("Prediction.")
Y = unet_model.predict(X, verbose=True)

probability_images = list()
for i in range(number_of_classification_labels):
    if verbose:
        print("Reconstructing image", i)
    prob_image = ants.from_numpy_like(np.squeeze(Y[0,:,:,:,i]), image_warped)
    if i==0:
        ants.image_write(prob_image, "/tmp/prob_image_warped.nii.gz")
    # probability_images.append(xfrm_inv.apply_to_image(prob_image, image))
    probability_images.append(prob_image)

for i in range(number_of_classification_labels):
    ants.image_write(probability_images[i], "/tmp/prob_"+str(i)+".nii.gz")

image_matrix = ants.image_list_to_matrix(probability_images, image * 0 + 1)
segmentation_matrix = np.argmax(image_matrix, axis=0)
segmentation_image = ants.matrix_to_images(
    np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

ants.image_write(segmentation_image, "./predicted_segmentation.nii.gz")
