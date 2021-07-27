import numpy as np
import os

from algo import *

from nilearn import image,plotting
import nibabel as nib

# define path
root_beta_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_predict/reorganized_results'

# read data
beta_fa = nib.load(os.path.join(root_beta_path,'train_fa','beta_raw.nii.gz'))
beta_fa = beta_fa.get_data().flatten()
beta_sa = nib.load(os.path.join(root_beta_path,'train_sa','beta_uncorrect.nii.gz'))
beta_sa = beta_sa.get_data().flatten()

# beta_data manipulation for prediction
    # note in beta_fa and beta_sa, the nan data are converted to 0 for surface plot
common_beta = abs(beta_fa)+abs(beta_sa)
diff_beta_fa = abs(beta_fa)-abs(beta_sa)  # diff denotes how
diff_beta_sa = -1*diff_beta_fa

## retrieve the orientaition of voxels
common_beta_fa = vectorized_back(beta_fa,common_beta)
common_beta_sa = vectorized_back(beta_sa,common_beta)

diff_beta_fa_v = vectorized_back(beta_fa,diff_beta_fa)
diff_beta_sa_v = vectorized_back(beta_sa,diff_beta_sa)

save_path = os.path.join(root_beta_path,'common_diff_map')
file_dic = {'common_beta_fa':common_beta_fa,
            'common_beta_sa':common_beta_sa,
            'diff_beta_fa':diff_beta_fa_v,
            'diff_beta_sa':diff_beta_sa_v}

for key,value in file_dic.items():
    np.save(os.path.join(save_path,key),value)
