import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from algo import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import LeaveOneOut,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import nibabel as nib

# define data path
path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/train_test_data'
beta_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_predict/reorganized_results/common_diff_map'

# read fa train and test data, also test on sa
train_X_fa = np.load(os.path.join(path,'train_X_fa.npy'))
test_X_fa = np.load(os.path.join(path,'test_X_fa.npy'))
train_Y_fa = np.load(os.path.join(path,'train_Y_fa.npy'))
test_Y_fa = np.load(os.path.join(path,'test_Y_fa.npy'))
X_sa = np.load(os.path.join(path,'X_sa.npy'))
Y_sa = np.load(os.path.join(path,'Y_sa.npy'))

mask_NaN = np.load(os.path.join(path,'mask_NaN.npy'))

beta_common = np.load(os.path.join(beta_path,'common_beta_fa.npy'))
beta_diff = np.load(os.path.join(beta_path,'diff_beta_fa.npy'))

beta_common = beta_common[mask_NaN]
beta_diff = beta_diff[mask_NaN] # reduce dimention to 140000 to eliminate Nan

### PCR train on fa #######
# scale data
train_X_fa = scale(train_X_fa)
test_X_fa = scale(test_X_fa)
X_sa = scale(X_sa)

# reduce dimention with PCA
pca = PCA() # reduce dimentions to the number of observations
pca.fit(train_X_fa)
train_X_fa = pca.transform(train_X_fa)
test_X_fa = pca.transform(test_X_fa)
X_sa = pca.transform(X_sa)

beta_common = pca.transform(beta_common.reshape(1,-1))
beta_diff = pca.transform(beta_diff.reshape(1,-1))

# build a logitic regression model
C_best = 0.05 # obtained from last prediction
clf = LogisticRegression(C=C_best)
clf.fit(train_X_fa,train_Y_fa)

## predicting using common beta
clf.coef_ = beta_common
