import h5py

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/DeepLesion'
base_img_dir = base_dir + '/deep_lesion'

patient_df = pd.read_csv(base_dir + '/DL_info.csv')
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir,
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)
patient_df['Radius'] = patient_df['Lesion_diameters_Pixel_'].map(lambda x: float(x.split(', ')[0]))
print('Loaded', patient_df.shape[0], 'cases')

# patient_df.sample(3)

patient_df['exists'] = patient_df['kaggle_path'].map(os.path.exists)
patient_df = patient_df[patient_df['exists']].drop('exists', 1)
# extact the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
print('Found', patient_df.shape[0], 'patients with images')


image_name = patient_df['File_name']
label = patient_df['Coarse_lesion_type']
bbox = patient_df['bbox']

labeled_idxs = np.where(label != -1)[0]

image_name = image_name[labeled_idxs]
label = label[labeled_idxs]
bbox = bbox[labeled_idxs]


x_train, x_test, y_train, y_test, bbox_train, bbox_test = train_test_split(image_name,
                                                                           label, bbox,
                                                                           test_size=0.1,
                                                                           random_state=2018)

np.savez_compressed('deeplesion_train', x=x_train, y=y_train, bbox=bbox_train)
np.savez_compressed('deeplesion_test', x=x_test, y=y_test, bbox=bbox_test)

print()







