import scipy.io as sio
import spectral as sp
import numpy as np
mat_in=sio.loadmat('salinas_in.mat')
m_in=mat_in['salinasA_corrected']
mat_gt=sio.loadmat('salinas_gt.mat')
m_gt=mat_gt['salinasA_gt']
#gt=sp.imshow(classes=m_gt,figsize=(5,5))
#def patch(X,ph,pw)
PATCH_SIZE=1
def mean_array(data):
    mean_arr = []
    for i in range(data.shape[0]):
        mean_arr.append(np.mean(data[i,:,:]))
    return np.array(mean_arr)
def Patch(data,height_index,width_index):
    transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    #print patch.shape
    mean = mean_array(transpose_array)
    mean_patch = []
    for i in range(patch.shape[0]):
        mean_patch.append(patch[i] - mean[i])
    mean_patch = np.asarray(mean_patch)
    patch = mean_patch.transpose((1,2,0))
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    print(patch.shape)
    #return patch