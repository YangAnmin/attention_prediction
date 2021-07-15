import numpy as np
import os

from algo import *

from nilearn import image,plotting
import nibabel as nib

from statsmodels.stats.multitest import fdrcorrection

# data path
root_path = '/users/anmin/desktop/results_duplicate/'
save_path = '/users/anmin/working/attention_results/plots'

num_voxels = {}

present_threshold = str(round((1 - 149270/902629),2)*100)+'%'

### plot beta map(uncorrected) #######
beta_path_fa = os.path.join(root_path,'train_fa','fa_beta_nan_transferred.nii.gz')
plot=plotting.view_img_on_surf(beta_path_fa, threshold=present_threshold, surf_mesh='fsaverage',title='fa_beta_map')
plot.save_as_html(os.path.join(save_path,'train_fa','fa_raw.html'))

beta_path_sa = os.path.join(root_path,'train_sa','sa_beta_nan_transferred.nii.gz')
plot=plotting.view_img_on_surf(beta_path_sa, threshold=present_threshold, surf_mesh='fsaverage',title='fa_beta_map')
plot.save_as_html(os.path.join(save_path,'train_sa','sa_raw.html'))

beta_fa = nib.load(beta_path_fa)
beta_fa = beta_fa.get_data()
num_voxels['feature_num'] = beta_fa.size
num_voxels['fa_with_value'] = np.sum(beta_fa!=0)

beta_sa = nib.load(beta_path_sa)
beta_sa = beta_sa.get_data()
num_voxels['sa_with_value'] = np.sum(beta_sa!=0)

### plot beta map(bonferroni corrected) #######
fa_p_path = os.path.join(root_path,'train_fa','boot_fa_z.npy')
sa_p_path = os.path.join(root_path,'train_sa','boot_sa_z.npy')

fa_p_array_full = np.load(fa_p_path)
sa_p_array_full = np.load(sa_p_path)
mask_nan = np.load(os.path.join(root_path,'mask_NaN.npy'))

# only voxels not Nan are preceeded with p-value correction
fa_p_array_masked = fa_p_array_full[mask_nan]
sa_p_array_masked = sa_p_array_full[mask_nan]

p_threshold = 0.05/size(fa_p_array_masked)

# fa p mask
mask1_fa = (fa_p_array_masked>(1-p_threshold/2))
mask2_fa = (fa_p_array_masked<(p_threshold/2))
is_above_t_fa = np.logical_or(mask1_fa,mask2_fa)
mask_t_fa = mask_back(mask_nan,is_above_t_fa,mask_type='beta')

# sa p mask
mask1_sa = (sa_p_array_masked>(1-p_threshold/2))
mask2_sa = (sa_p_array_masked<(p_threshold/2))
is_above_t_sa = np.logical_or(mask1_sa,mask2_sa)
mask_t_sa = mask_back(mask_nan,is_above_t_sa,mask_type='beta')

# beta value
beta_matrix_fa = nib.load(beta_path_fa)
ori_shape = np.array(beta_matrix_fa.get_data()).shape
affine = beta_matrix_fa.affine.copy()
hdr = beta_matrix_fa.header.copy()
beta_matrix_fa = beta_matrix_fa.get_data()

beta_matrix_sa = nib.load(beta_path_sa)
beta_matrix_sa = beta_matrix_sa.get_data()

mask_t_fa = mask_t_fa.reshape(ori_shape)
mask_t_sa = mask_t_sa.reshape(ori_shape)

beta_matrix_fa_cutoff = beta_matrix_fa*mask_t_fa # beta value passed correction
beta_matrix_sa_cutoff = beta_matrix_sa*mask_t_sa

fa_mni = nib.Nifti1Image(beta_matrix_fa_cutoff,affine,hdr)
sa_mni = nib.Nifti1Image(beta_matrix_sa_cutoff,affine,hdr)

nib.save(fa_mni,'/users/anmin/working/attention_results/analized_data/fa_beta_bonferroni.nii.gz')
nib.save(sa_mni,'/users/anmin/working/attention_results/analized_data/sa_beta_bonferroni.nii.gz')

# document surving number of voxels
num_voxels['fa_bonferroni'] = np.sum(mask_t_fa==1)
num_voxels['sa_bonferroni'] = np.sum(mask_t_sa==1)

plot = plotting.view_img_on_surf(fa_mni, threshold=present_threshold, surf_mesh='fsaverage',title='fa_bonferroni')
plot.save_as_html(os.path.join(save_path,'train_fa','fa_corrected_bonferroni.html'))

plot=plotting.view_img_on_surf(sa_mni, threshold=present_threshold, surf_mesh='fsaverage',title='sa_bonferroni')
plot.save_as_html(os.path.join(save_path,'train_sa','sa_corrected_bonferroni.html'))

### plot beta map with FDR correction #######
from statsmodels.stats.multitest import fdrcorrection

fa_p_array_fdr = fdrcorrection(fa_p_array_masked)[1]
sa_p_array_fdr = fdrcorrection(sa_p_array_masked)[1]
p_threshold = 0.05

# fa p mask
mask1_fa = (fa_p_array_fdr>(1-p_threshold/2))
mask2_fa = (fa_p_array_fdr<(p_threshold/2))
is_above_t_fa = np.logical_or(mask1_fa,mask2_fa)
mask_t_fa = mask_back(mask_nan,is_above_t_fa,mask_type='beta')

# sa p mask
mask1_sa = (sa_p_array_fdr>(1-p_threshold/2))
mask2_sa = (sa_p_array_fdr<(p_threshold/2))
is_above_t_sa = np.logical_or(mask1_sa,mask2_sa)
mask_t_sa = mask_back(mask_nan,is_above_t_sa,mask_type='beta')

mask_t_fa = mask_t_fa.reshape(ori_shape)
mask_t_sa = mask_t_sa.reshape(ori_shape)

beta_matrix_fa_cutoff = beta_matrix_fa*mask_t_fa # beta value passed correction
beta_matrix_sa_cutoff = beta_matrix_sa*mask_t_sa

fa_mni = nib.Nifti1Image(beta_matrix_fa_cutoff,affine,hdr)
sa_mni = nib.Nifti1Image(beta_matrix_sa_cutoff,affine,hdr)

nib.save(fa_mni,'/users/anmin/working/attention_results/analized_data/fa_beta_fdr.nii.gz')
nib.save(sa_mni,'/users/anmin/working/attention_results/analized_data/sa_beta_fdr.nii.gz')

# document surving number of voxels
num_voxels['fa_fdr'] = np.sum(mask_t_fa==1)
num_voxels['sa_fdr'] = np.sum(mask_t_sa==1)

plot = plotting.view_img_on_surf(fa_mni, threshold=present_threshold, surf_mesh='fsaverage',title='fa_fdr')
plot.save_as_html(os.path.join(save_path,'train_fa','fa_corrected_fdr.html'))

plot = plotting.view_img_on_surf(sa_mni, threshold=present_threshold, surf_mesh='fsaverage',title='sa_fdr')
plot.save_as_html(os.path.join(save_path,'train_sa','sa_corrected_fdr.html'))
