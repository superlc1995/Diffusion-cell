from skimage import morphology

import numpy as np
from PIL import Image

file_list = np.loadtxt('/home/lchen/tensorf/Dataset-BRCA-M2C/brca_ds_train.txt', dtype = 'str')

for file in file_list:
    # print(file)
    img = np.array(Image.open('/home/lchen/tensorf/Dataset-BRCA-M2C/images/' + file))
    out_den = np.zeros([int(img.shape[0]/2) + 1, int(img.shape[1]/2) + 1, 3])

    for type in [1,2,3]:
        
        # kde_den = np.zeros( [ int(img.shape[0]/2), int(img.shape[1]/2), 3] )
        labels = np.loadtxt( '/home/lchen/tensorf/Dataset-BRCA-M2C/labels/' + file.split('.png')[0] + '_gt_class_coords.txt' )
        labels = labels[labels[:,2] == type]
        labels = labels/2

        point_den = np.zeros([int(img.shape[0]/2) + 1, int(img.shape[1]/2) + 1])

        point_den[labels[:,0].astype(int), labels[:,1].astype(int)] = 1

        out_den[:,:,(type - 1)] = morphology.dilation( point_den , morphology.square(3))

    np.save('point_den/' + file.split('.png')[0] + '.npy', out_den )