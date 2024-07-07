from sklearn.mixture import GaussianMixture
import numpy as np
from PIL import Image
import scipy.io

file_list = np.loadtxt('BRCA_dataset/brca_ds_train.txt', dtype = 'str')

for file in file_list:

    # data = scipy.io.loadmat('/data07/shared/chenli/gen_diff/PSU_dataset/img' + str(idx) + '/img' + str(idx) + '_detection.mat')['detection']
    img = np.array(Image.open('BRCA_dataset/images/' + file))
    gmm_den = np.zeros( [ int(img.shape[0]/2) + 1, int(img.shape[1]/2)+1, 3] )
    for classes in [1,2,3]:
        data = np.loadtxt( 'BRCA_dataset/labels/' + file.split('.png')[0] + '_gt_class_coords.txt' )

        am = data[:,0].copy()
        data[:,0] = data[:,1]
        data[:,1] = am

        data = data[data[:,2] == classes]

        data = data/2 
        data = data[:,:2]
        if len(data) > 2:
    
            min_bic = 1e30
            best_cluster = 0
            for num_clust in np.arange(1,11):
                if len(data) > num_clust :
                    gm = GaussianMixture(n_components=num_clust, random_state=0).fit(data)
                    if gm.bic(data) < min_bic:
                        min_bic = gm.bic(data)
                        best_cluster = num_clust

            gm = GaussianMixture(n_components=best_cluster, random_state=0).fit(data)

            test_points = []
            for y in range(int(img.shape[0] / 2)):
                for x in range(int(img.shape[1] / 2)):
                    test_points.append([x,y])

            test_points = np.array(test_points)
            gmm_den[:int(img.shape[0] / 2),:int(img.shape[1] / 2),classes - 1] = np.reshape(np.exp(gm.score_samples(test_points)) , [int(img.shape[0]/2) , int(img.shape[1]/2) ]) 

    np.save('gmm_den/' + file.split('.png')[0] + '.npy', gmm_den )