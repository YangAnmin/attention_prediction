from nilearn import image

import numpy as np
from random import sample
import os

from algo import *

# define path and subjects for analysis
root_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/data'

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

# deside feature size
name = train_sub_name[0]
path_fa = os.path.join(root_path,name,'1st_fa')
test_array = image.get_data(os.path.join(path_fa,'beta_0001.nii')).flatten()
feature_num = len(test_array)

# define not-NaN features
X_fa = X_matrix(sub_name,root_path,'fa',len(test_array))
X_sa = X_matrix(sub_name,root_path,'sa',len(test_array))
mask_fa = nan_mask(X_fa)
mask_sa = nan_mask(X_sa)
mask_NaN = mask_fa*mask_sa   # take the union of both masks

np.save('/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/train_test_data/mask_NaN',mask_NaN)

### stack train_X ang train_Y
train_X_fa = X_matrix(train_sub_name,root_path,'fa',len(test_array))[:,mask_NaN]
train_Y_fa = Y_matrix(train_X_fa)

train_X_sa = X_matrix(train_sub_name,root_path,'sa',len(test_array))[:,mask_NaN]
train_Y_sa = Y_matrix(train_X_sa)

## stack test_X and text_Y
test_X_fa = X_matrix(test_sub_name,root_path,'fa',len(test_array))[:,mask_NaN]
test_Y_fa = Y_matrix(test_X_fa)

test_X_sa = X_matrix(test_sub_name,root_path,'sa',len(test_array))[:,mask_NaN]
test_Y_sa = Y_matrix(test_X_sa)

## stack intact sa matrix
X_fa = X_fa[:,mask_NaN]
X_sa = X_sa[:,mask_NaN]

Y_sa = Y_matrix(X_sa)
Y_fa = Y_matrix(X_fa)

# save train and test data
save_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/train_test_data'

file_dic = {'X_fa':X_fa,'X_sa':X_sa,'Y_fa':Y_fa,'Y_sa':Y_sa,
            'train_X_fa':train_X_fa,'train_X_sa':train_X_sa,'test_X_fa':test_X_fa,'test_X_sa':test_X_sa,
            'train_Y_fa':train_Y_fa,'train_Y_sa':train_Y_sa,'test_Y_sa':test_Y_sa,'test_Y_fa':test_Y_fa}

for key,value in file_dic.items():
    np.save(os.path.join(save_path,key),value)
