import os
import numpy as np
import nibabel as nib
from nilearn import image
from sklearn.preprocessing import scale
from scipy.stats import norm
import time

def X_matrix(sub_name,path,model_type,feature_num):
    """
    generate X input matrix for predictor
    2D array (subjects,features)

    Parameter
    ---------
    sub_name: list
        list of sujects' names
    path: str
        path containing files of every subject
    model_type: str
        'fa': model trained on feature-based attention
        'sa': model trained on spacial-based attention
    feature_num: int
        number of features, how many voxels in one subject

    Retrun
    ------
    X: ndarray
        2D array of feature matrix, (subjects,features)
    """

    X = np.zeros(feature_num)

    if model_type == 'fa':
        file_name = '1st_fa'
        for name in sub_name:
            path_temp = os.path.join(path,name,file_name)
            feature_array = image.get_data(os.path.join(path_temp,'beta_0001.nii')).flatten()
            conjunction_array = image.get_data(os.path.join(path_temp,'beta_0002.nii')).flatten()
            X = np.vstack((X,feature_array,conjunction_array))
    else:
        file_name = '1st_sa'
        for name in sub_name:
            path_temp = os.path.join(path,name,file_name)
            array1 = image.get_data(os.path.join(path_temp,'beta_0001.nii')).flatten()
            array2 = image.get_data(os.path.join(path_temp,'beta_0002.nii')).flatten()
            X = np.vstack((X,array1,array2))

    X = X[1:,:]

    return X

def Y_matrix(X):
    """
    generate Y input matrix for predictor
    1D array (labels)

    Parameter
    ---------
    X: int
        label number according to the shape of input X (how many subjects）

    Return
    ------
    Y: ndarray
        1D array of labels
        (0,1) periodically for fa where 0 denotes feature, 1 denotes conjunction
        (0,1,1) periodically for sa  where 0 denotes central, 1 denotes peripheral

    """
    length = X.shape[0]
    Y = np.array((0,1)*(int(length/2)))

    return Y

def nan_mask(matrix):
    """
    find features that contain NaN and return a NaN mask

    Parameter
    ---------
    matrix: ndarray
        2D matrix (subjects,features)(m,n)

    Retrun
    ------
    mask: ndarray
        1D array, if one value esistes in one feature, that feature is eliminated
        represented in bool value
    """
    is_nan = np.isnan(matrix)
    is_nan_array = is_nan.sum(axis=0) # is a NaN in this feature, then the sum is not 0
    mask = (is_nan_array == 0)

    return mask

def mask_back(mask_matrix,value_matrix,mask_type):
    """
    back project value matrix to original matrix
    the shape of the matrix returned shoud be like mask matrix
    used for reduced beta coefficients and mni space

    Parameter
    ---------
    mask_matrix: ndarray
        matrix with bool value, 1D array
    value_matrix: ndarray
        matrix with real value, the size of which is smaller than mask matrix
    mask_type: string
        the type of mask_matrix
        'beta': mask the present beta value back to full PC size,
            in which case the masked out value will be filled with 0
        'mni': mask the reverse transformed beta value to full mni space,
            in which case the masked out value will be set as nan

    Return
    ------
    reshaped_matrix: ndarray
        projecting real value to original matrix

    """
    length_mask = len(mask_matrix)
    lis_mask = list(mask_matrix.flatten())
    lis_value = list(value_matrix.flatten())

    counter = 0
    for i in range(length_mask):
        if lis_mask[i]:
            lis_mask[i] = lis_value[counter]
            counter += 1
        else:
            if mask_type == 'mni':
                lis_mask[i] = np.nan
            else:
                continue
    reshaped_matrix = np.array(lis_mask)

    return reshaped_matrix

def bootstrap_data(original_data):
    """
    bootstrap from orininal data to create a new data set, sample without replacement

    Parameter
    ---------
    original_data: ndarray
        2D matrix, (subjects,features) (m,n)

    Return
    ------
    new_matrix: ndarray
        new data set by bootstrap, sample with repalcement
        2D matrix, (subjects,features) (m,n)
    """
    select_range = original_data.shape[0]
    indexes = np.random.choice(a=select_range,size=select_range)

    new_matrix = np.zeros(original_data.shape[1])
    for index in indexes:
        new_matrix = np.vstack((new_matrix,original_data[index]))

    new_matrix = new_matrix[1:,:]

    return new_matrix

def p_transfer(data_matrix):
    """
    transfer beta map to p-map, from beta to p value
    return the p value of the observed array

    Parameter
    ---------
    data_matrix: ndarray
        beta map, 2D array, (m,n)
        m: number of subjects
        n: number of features, denoting number of voxels

    Return
    ------
    p_map: ndarray
        1D array, (n,)
        every element denotes the p value of that voxel
    """
    start = time.time()

    length = data_matrix.shape[1] # how many features
    data_matrix = scale(data_matrix).flatten() # convert to Z score
    data_matrix = data_matrix[-length:] # only the real z-score instead of bootstrapped scores are considered

    mid = time.time()
    print(f'Converting Z scores is complete, using time {mid-start} seconds.')

    p_map = [norm.cdf(value) for value in data_matrix]
    p_map = np.array(p_map)

    end = time.time()
    print(f'Calculation of p value is complete, using time {end-mid} seconds.')

    return p_map

def to_mni(data_array):
    """
    convert array(beta) to mni space
    specific for the analysis

    Parameter
    ---------
    data_array: ndarray
        1D array, contraining beta value

    Return
    ------
    mni_data: Nifti1Image
        NiftiImage, with space and afine specific for this analysis
    """
    # acquire sample parameters
    sample_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/data/S0001/1st_fa/beta_0001.nii'
    sample_nii = nib.load(sample_path)
    ori_shape = np.array(sample_nii.get_data()).shape
    affine = sample_nii.affine.copy()
    hdr = sample_nii.header.copy()

    del sample_nii

    mni_data = nib.Nifti1Image(data_array.reshape(ori_shape),affine,hdr)

    return mni_data

def p_threshold(path_beta,path_p,threshold):
    """
    define a p_threshold of activation
    the value below the threshold is set to 0

    Parameter
    ---------
    path_beta: str
        the path of nii file, beta value
    path_p: str
        the path of nii file, p value
    threshold: float
        the threshold of p value, range is [0,1]

    Return
    ------
    masked_array: ndarray
        the value below threshold is set to 0
    """
    # basic parameters of nii file
    nii_map_beta = nib.load(path_beta)
    activation_map = nii_map_beta.get_data()

    nii_map_p = nib.load(path_p)
    p_map = nii_map_p.get_data()

    mask1 = (p_map>(1-threshold/2))
    mask2 = (p_map<(threshold/2))
    is_above_t = np.logical_or(mask1,mask2)
    corrected_map = activation_map*is_above_t

    return corrected_map

def vectorized_back(ori_array,modified_array):
    """
    the modified_array is manipulated by abs,and hense a scalar, lossing orientation information as a vector
    the orientation information is preserved in ori_array, derterming whether a scalar is positive or negative
    this function change a scalar back to vector

    Parameter
    ---------
    ori_array: ndarray
        1-D array, the original array storing orientation information
    modified_array: ndarray
        1-D array, the modified array lossing orientation information

    Return
    ------
    vectorized_array: ndarray
        1-D array, each item is a vector, storing both orientaiton information and modified value
    """
    mask = (ori_array<0)
    input_array = list(modified_array)

    for i in range(mask.size):
        if mask[i]: # the original value is negtive
            input_array[i] = -1*input_array[i]

    vectorized_array = np.array(input_array)

    return vectorized_array
