import numpy as np
import os
from algo import *

import nibabel as nib

# fetch data
root_path = '/nfs/s2/userhome/yanganmin/workingdir/attention_data_complete/results'
fa_bootstrap = np.load(os.path.join(root_path,'train_fa','beta_bootstrap.npy'))
sa_bootstrap = np.load(os.path.join(root_path,'train_sa','beta_bootstrap.npy'))

fa = nib.load(os.path.join(root_path,'train_fa','beta_raw.nii.gz'))
fa = np.array(fa.get_data()).flatten()

sa = nib.load(os.path.join(root_path,'train_sa','beta_uncorrect.nii.gz'))
sa = np.array(sa.get_data()).flatten()

# convert beta map to z_value
fa_bootstrap = np.vstack((fa_bootstrap,fa))
boot_fa_z = p_transfer(fa_bootstrap)
del fa_bootstrap
np.save(os.path.join(root_path,'train_fa','boot_fa_z'),boot_fa_z)

sa_bootstrap = np.vstack((sa_bootstrap,sa))
boot_sa_z = p_transfer(sa_bootstrap)
del sa_bootstrap
np.save(os.path.join(root_path,'train_fa','boot_sa_z'),boot_sa_z)

# convert nan value to zero
boot_fa_z = np.nan_to_num(boot_fa_z)
boot_sa_z = np.nan_to_num(boot_sa_z)

fa_p_mni = to_mni(boot_fa_z)
sa_p_mni = to_mni(boot_sa_z)

nib.save(fa_p_mni, os.path.join(root_path,'train_fa','fa_p_mni_nan_transferred.nii.gz'))
nib.save(sa_p_mni, os.path.join(root_path,'train_sa','sa_p_mni_nan_transferred.nii.gz'))

 
