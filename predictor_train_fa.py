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

# read fa train and test data, also test on sa
train_X_fa = np.load(os.path.join(path,'train_X_fa.npy'))
test_X_fa = np.load(os.path.join(path,'test_X_fa.npy'))
train_Y_fa = np.load(os.path.join(path,'train_Y_fa.npy'))
test_Y_fa = np.load(os.path.join(path,'test_Y_fa.npy'))
X_sa = np.load(os.path.join(path,'X_sa.npy'))
Y_sa = np.load(os.path.join(path,'Y_sa.npy'))

mask_NaN = np.load(os.path.join(path,'mask_NaN.npy'))

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

# linear regression with L1 regulation

## grid search for best hyper-Parameter
tuned_parameters = [{'C': np.arange(0.05,1.05,0.05)}]

clf = GridSearchCV(LogisticRegression(penalty='l1',solver='liblinear'),tuned_parameters,cv=10,scoring='f1')
clf.fit(train_X_fa, train_Y_fa)

C_best = clf.best_params_['C']

clf = LogisticRegression(C=C_best) # C value determined by grid search


loo = LeaveOneOut()
LR_beta = []
## train LR to find low beta PCs
for train_index, test_index in loo.split(train_X_fa):
    clf.fit(train_X_fa[train_index], train_Y_fa[train_index])
    LR_beta.append(clf.coef_[0])

LR_beta_matrix = np.zeros(len(LR_beta[0]))
for array in LR_beta:
    LR_beta_matrix = np.vstack((LR_beta_matrix,array))
LR_beta_matrix = LR_beta_matrix[1:,:]
LR_beta_iszero = LR_beta_matrix.mean(axis=0) # in this complete data set, no zero beta
                                            # a threshold is always an option, abs(beta) < 0.01 for instance

### rule out low-beta pcs
beta_cutoff = 0.01
mask_beta = (abs(LR_beta_iszero) > beta_cutoff)
train_X_fa = train_X_fa[:,mask_beta]
test_X_fa = test_X_fa[:,mask_beta]
X_sa = X_sa[:,mask_beta]

## train lasso with leave-one-out and independent hold-out data
    ### also used gird search for best C parameter, turned out the same regardless of the reduced featrues
beta_pool = []
leave_out_result = [] # tupple inside (train_accuracy,val_accuracy)
test_result = []
sa_result = []

for train_index, test_index in loo.split(train_X_fa):
    clf.fit(train_X_fa[train_index], train_Y_fa[train_index])

    coef = clf.coef_[0]
    coef_modified = mask_back(mask_beta,coef,mask_type='beta')
    beta_fa = pca.inverse_transform(coef_modified)
    beta_pool.append(beta_fa)

    val_accuracy =  clf.score(train_X_fa[test_index], train_Y_fa[test_index])
    train_accuracy = clf.score(train_X_fa[train_index],train_Y_fa[train_index])
    leave_out_result.append((train_accuracy,val_accuracy))

    test_result.append(clf.score(test_X_fa,test_Y_fa))
    sa_result.append(clf.score(X_sa,Y_sa))

### predict accuracy
train_acc_array = [i[0] for i in leave_out_result]
val_acc_array = [i[1] for i in leave_out_result]

### save prediction results
train_result = np.array(train_acc_array)
val_result = np.array(val_acc_array)
test_result = np.array(test_result)
sa_result = np.array(sa_result)
result_matrix = np.c_[train_result,val_result,test_result,sa_result]
df_result = pd.DataFrame(result_matrix, columns=['train','val','test','sa'])

save_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/results'
df_result.to_csv(os.path.join(save_path,'train_fa','prediction_result.csv'))

### average beta
avg_beta = np.zeros(beta_pool[0].shape[0])
for pool in beta_pool:
    avg_beta = np.vstack((avg_beta,pool))
avg_beta = avg_beta[1:,:]
avg_beta = avg_beta.mean(axis=0)
avg_beta = scale(avg_beta)

### back-project beta map to mni space
coef_mni = mask_back(mask_NaN,avg_beta,mask_type='mni')
coef_mni = np.nan_to_num(coef_mni) # convert nan to zero
nif_beta_raw = to_mni(coef_mni)
nib.save(nif_beta_raw, os.path.join(save_path,'train_fa','fa_beta_nan_transferred.nii.gz'))

### bootstrap for robust beta value #######
# clear cache for bootstrap
del coef_mni,avg_beta,beta_fa,nif_beta_raw,LR_beta_matrix,pool,X_sa

# bootstrap procedure
sample_size = 1000
boot_coef = np.zeros(mask_NaN.shape[0])

concat_data = np.c_[train_X_fa,train_Y_fa]

for iteration in range(sample_size):
    bootstraped_data = bootstrap_data(concat_data)
    X = bootstraped_data[:,:-1] # use only the features that beta value exceed cutoff value
    Y = bootstraped_data[:,-1]              # based on the original dataset

    clf.fit(X,Y)
    coef = clf.coef_[0]
    coef_modified = mask_back(mask_beta,coef,mask_type='beta')
    inverse_coef= pca.inverse_transform(coef_modified)
    coef_mni = mask_back(mask_NaN,inverse_coef,mask_type='mni')
    boot_coef = np.vstack((boot_coef,coef_mni))
boot_coef = boot_coef[1:,:]
np.save('/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/results/train_fa/beta_bootstrap',boot_coef)
