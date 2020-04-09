import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
import imageio
from skimage.transform import resize

def get_square_crop(img, base_size=256, crop_size=256):
    res = img
    height, width = res.shape
    if height < base_size:
        diff = base_size - height
        extend_top = int(diff / 2)
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
        height = base_size

    if width < base_size:
        diff = base_size - width
        extend_top = int(diff / 2)
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)
        width = base_size

    crop_y_start = int((height - crop_size) / 2)
    crop_x_start = int((width - crop_size) / 2)
    res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]
    return res

def normalize_to_plus_minus_one(img):
    img_zero = img - np.amin(img)
    img_one = img_zero / np.amax(img_zero)
    img_one_one = img_one * 2.0 - 1.0
    return img_one_one

def square_complete(image):
    if image.ndim == 2:
        max_dim = max(image.shape)
        min_dim = min(image.shape)
        add_zeros = int((max_dim - min_dim)/2)
        height, width = image.shape
        if height > width:
            image_new = np.pad(image, [(0, 0), (add_zeros, add_zeros)], mode='constant')
        else:
            image_new = np.pad(image, [(add_zeros, add_zeros), (0, 0)], mode='constant')
        return image_new
    else:
        raise Exception('Input image should be 2D !!!')
        
def process_cheXpert(label_df, base_path, Max_num, n_class, output_dir, label_dir, resolution=256):
    new_labels = np.empty((Max_num, n_class))
    Image_id_list = []
    for Image_ID in range(Max_num):
        image_path = os.path.join(base_path, label_df['Path'][Image_ID])
        image_index = label_df['Image Index'][Image_ID]
        image = imageio.imread(image_path)
        if image.ndim != 2:
            image = image[:,:,0]

        image_resize = resize(image, (resolution, resolution), anti_aliasing=True)
        image_resize = normalize_to_plus_minus_one(image_resize)
        image_resize = np.clip(np.rint((image_resize + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
        image_name = str(image_index).zfill(6) + '.png'
        imageio.imwrite(os.path.join(output_dir, image_name), image_resize)

        row = label_df.loc[label_df['Image Index'] == image_index]
        Image_id_list.append(image_name)
        pathologies = row.columns[1:1+n_class]
        for j, pathology in enumerate(pathologies):
            value = row[pathology]
            new_labels[Image_ID,j,] = value

    pathology_dict = dict(zip(Image_id_list, new_labels))
    label_path = os.path.join(label_dir, 'label_CheXpert.npy')
    if len(pathology_dict) == Max_num:
        np.save(label_path, pathology_dict)
    else:
        raise Exception('Imcomplete label !!!')

base_path = '/media/tianyu.han/mri-scratch/DeepLearning/Stanford_MIT_CHEST'
csv_path = '/media/tianyu.han/mri-scratch/DeepLearning/CheXpert_Dataset/label.csv'
output_dir = '/media/tianyu.han/mri-scratch/DeepLearning/CheXpert_Dataset/images_256/images'
label_dir = '/media/tianyu.han/mri-scratch/DeepLearning/CheXpert_Dataset/images_256/labels'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

label_df = pd.read_csv(csv_path)
process_cheXpert(label_df, base_path, len(label_df), 14, output_dir, label_dir, resolution=256)