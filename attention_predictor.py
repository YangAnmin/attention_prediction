from nilearn import image

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut

import PCRegression

import numpy as np
from random import sample
import os

from algo import *

# define path and subjects for analysis
root_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_predict/data'

sub_name = os.listdir(root_path)

substract_name = []  # subjects that should be ruled out from further analysis
                     # for intensive head motion
for name in sub_name:
    file_num = len(os.listdir(os.path.join(root_path,name)))
    if file_num != 6:
        substract_name.append(name)

sub_name = list(set(sub_name)-set(substract_name))

sub_num = len(sub_name)
test_data_size = round(sub_num/3)
test_sub_name = sample(sub_name,test_data_size)
train_sub_name = list(set(sub_name)-set(test_sub_name))



### model_fa #######

# pool data matrix
## train X_fa and train_Y_fa
### deside feature size
name = train_sub_name[0]
path_fa = os.path.join(root_path,name,'1st_fa')
test_array = image.get_data(os.path.join(path_fa,'beta_0001.nii')).flatten()
feature_num = len(test_array)

### define not-NaN features
X_fa = X_matrix(sub_name,root_path,'fa',len(test_array))
X_sa = X_matrix(sub_name,root_path,'sa',len(test_array))
mask_fa = nan_mask(X_fa)
mask_sa = nan_mask(sa)
mask_NaN = mask_fa*mask_sa   # take the union of both masks

### stack train_X ang train_Y
train_X_fa = X_matrix(train_sub_name,root_path,'fa',len(test_array))
train_X_fa = shink_nan(train_X_fa)
train_Y_fa = Y_matrix(train_X_fa,'fa')


## stack test_X and text_Y
test_X_fa = X_matrix(test_sub_name,root_path,'fa',len(test_array))
test_Y_fa = Y_matrix(test_X_fa,'fa')

## stack intact sa matrix
X_sa = X_matrix(sub_name,root_path,'sa',len(test_array))
Y_sa = Y_matrix(X_sa,'sa')

# PCR model
## determine PCA components

loo = LeaveOneOut()
loo.get_n_splits(train_X_fa)
