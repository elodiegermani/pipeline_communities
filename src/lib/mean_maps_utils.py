from lib import louvain_utils
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt
from glob import glob
from nilearn import plotting, image, masking, input_data, datasets, glm
import nibabel as nib
import numpy as np
import pandas as pd
import os
from nibabel.processing import resample_from_to


def compute_mean_maps(data_path, repo_path, contrast, partition, data_type):
    if not os.path.exists(f'{repo_path}/figures/mean_img_{data_type}_community_0_con_{contrast}.nii') or not os.path.exists(f'{repo_path}/figures/mean_img_{data_type}_pipeline_fsl-5-0-0_con_{contrast}.nii'):
        if data_type == 'group':
            data_fpath = f'{data_path}/group-1_{contrast}_*_tstat.nii*'

        elif data_type == 'sub':
            data_fpath = f'{data_path}/sub-100206_{contrast}_*_tstat.nii*'
        
        subject = [img.split('/')[-1].split('_')[2].split('-')[0].upper() +','+img.split('/')[-1].split('_')[2].split('-')[1]+','+img.split('/')[-1].split('_')[2].split('-')[2] +',' + img.split('/')[-1].split('_')[2].split('-')[3] for img in sorted(glob(data_fpath))]
        
        target = datasets.load_mni152_gm_template(4)
        mask = louvain_utils.compute_intersection_mask(data_fpath)
        masker = input_data.NiftiMasker(
                mask_img=mask)

        print('Computing mean image...')
        for community in np.unique(list(partition.values())):
            pipelines = [sub for i, sub in enumerate(subject) if partition[i]==community]
            print('Pipelines belonging to', community, ':', pipelines)
            mean_data=[]
            for pi in pipelines:
                data = []
                soft = pi.split(',')[0].lower()
                f = pi.split(',')[1]
                p = pi.split(',')[2]
                h = pi.split(',')[3]

                data_fpath = sorted(glob(f'{data_path}/{data_type}-*_{contrast}_{soft}-{f}-{p}-{h}_tstat.nii'))

                for fpath in data_fpath:
                    resampled_gm = image.resample_to_img(
                                nib.load(fpath),
                                target,
                               interpolation='continuous')

                    data.append(resampled_gm)

                maskdata = masker.fit_transform(data)
                meandata = np.mean(maskdata, 0)
                mean_img = masker.inverse_transform(meandata)

                nib.save(mean_img, f'{repo_path}/figures/mean_img_{data_type}_pipeline_{soft}-{f}-{p}-{h}_con_{contrast}.nii')

                mean_data.append(mean_img)

            mask_mean_data = masker.fit_transform(mean_data)
            meandata = np.mean(mask_mean_data, 0)
            mean_img = masker.inverse_transform(meandata)

            nib.save(mean_img, f'{repo_path}/figures/mean_img_{data_type}_community_{community}_con_{contrast}.nii')
    else:
        print('Mean image already computed.')

        
def compute_unilateral_masks(masks):
    '''
    Compute the global mask of 2 ROI and specific mask for each hemisfer. 

    Parameters:
        - masks, list of Nifti1Image: masks of the two ROI

    Returns:
        - mask, Nifti1Image: global mask
        - mask_right, Nifti1Image: right hemisfer mask
        - mask_left, Nifti1Image: left hemisfer mask
    '''
    mask_data=masks[0].get_fdata() + masks[1].get_fdata()

    mask = image.new_img_like(masks[0], mask_data, affine=masks[0].affine, copy_header=True)

    x_dim = mask_data.shape[0]
    x_center = int(x_dim/2)

    mask_data_left = mask_data.copy()
    mask_data_left[0:x_center,:,:] = 0
    mask_left = image.new_img_like(masks[0], mask_data_left, affine=masks[0].affine, copy_header=True)

    mask_data_right = mask_data.copy()
    mask_data_right[x_center:,:,:] = 0
    mask_right = image.new_img_like(masks[0], mask_data_right, affine=masks[0].affine, copy_header=True)

    return mask, mask_right, mask_left
        
def plot_mean_image(contrast, partition, repo_path, data_type):
    n_communities=len(np.unique(list(partition.values())))
    fig = plt.figure(figsize = (7 * n_communities, 14))
    gs = fig.add_gridspec(2, n_communities)
    
    fig.suptitle(f'{contrast.upper()}', size=28, fontweight='bold', backgroundcolor= 'black', color='white')
    
    for community in range(n_communities):
        mean_img = nib.load(f'{repo_path}/figures/mean_img_{data_type}_community_{community}_con_{contrast}.nii')
        ax = fig.add_subplot(gs[0, int(community)])

        disp = plotting.plot_glass_brain(mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                 cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig, axes=ax)
        disp.title(f'Community {community+1}', size=28, fontweight='bold')

        thresh_mean_img, threshold = glm.threshold_stats_img(mean_img, alpha=0.05, height_control='fdr',
                                                         two_sided=False)

        ax = fig.add_subplot(gs[1, int(community)])

        disp2 = plotting.plot_glass_brain(thresh_mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                 cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig, axes=ax)
        #disp2.title(f'Community {community+1}', size=28, fontweight='bold')

    fig.savefig(f'{repo_path}/figures/mean_maps_{data_type}_communities_{contrast}.png', dpi=300) 

def get_activated_voxels(contrast, partition, repo_path, data_type):
    atlas_roi = datasets.fetch_atlas_juelich('prob-2mm')
    lab =[31,32]
    masks = [image.index_img(atlas_roi.maps, lab_i) for lab_i in lab]
    mask, mask_right, mask_left = compute_unilateral_masks(masks)
    
    if contrast == 'left-foot' or contrast == 'left-hand':
        roi_mask = mask_left # Controlateral activation
    elif contrast =='right-foot' or contrast == 'right-hand':
        roi_mask = mask_right
    else:
        roi_mask = mask

    n_communities=len(np.unique(list(partition.values())))
    
    activ_voxels = []
    thresh_activ_voxels = []
    
    for community in range(n_communities):
        mean_img = nib.load(f'{repo_path}/figures/mean_img_{data_type}_community_{community}_con_{contrast}.nii')

        thresh_mean_img, threshold = glm.threshold_stats_img(mean_img, alpha=0.05, height_control='fdr',
                                                         two_sided=False)
        # Whole map
        img_data_act = np.nan_to_num(thresh_mean_img.get_fdata()) > 0
        n_activated_voxels = img_data_act.sum()
        activ_voxels.append(n_activated_voxels)
        
        # ROI 
        res_mask = resample_from_to(roi_mask, mean_img)
        res_mask_data = res_mask.get_fdata() > 1e-6
        mask_thresh_img = np.nan_to_num(thresh_mean_img.get_fdata() * res_mask_data.astype('int'))
        thresh_activ_voxels.append(np.count_nonzero(mask_thresh_img))
        
    df = pd.DataFrame({'Community': range(n_communities), 'Whole maps': activ_voxels, 'ROI': thresh_activ_voxels})
    df.to_csv(f'{repo_path}/figures/activated_voxels_{data_type}_communities_{contrast}.csv')
    
def plot_pipeline_maps(contrast, partition, subject, repo_path, data_type):
    communities = [[i for i in partition.keys() if partition[i]==p] for p in np.unique(list(partition.values())).tolist()]
    
    community = {}
    for k in range(len(communities)):
        community[k] = []
        for i, s in enumerate(subject):
            if i in communities[k]:
                community[k].append(s)
                
    atlas_roi = datasets.fetch_atlas_juelich('prob-2mm')
    lab =[31,32]
    masks = [image.index_img(atlas_roi.maps, lab_i) for lab_i in lab]
    mask, mask_right, mask_left = compute_unilateral_masks(masks)

    n_communities=len(community)
    fig = plt.figure(figsize = (7 * 8, 7 * n_communities))
    gs = fig.add_gridspec(n_communities, 10)

    fig2 = plt.figure(figsize = (7 * 8, 7 * n_communities))
    gs2 = fig.add_gridspec(n_communities, 10)

    for com in range(n_communities):
        for i, p in enumerate(community[com]):
            p_str = p.split(',')[0].lower() + '-' + p.split(',')[1]+ '-' + p.split(',')[2]+ '-' + p.split(',')[3]
            mean_img = nib.load(f'{repo_path}/figures/mean_img_{data_type}_pipeline_{p_str}_con_{contrast}.nii')
            ax = fig.add_subplot(gs[com, int(i)])
            
            if i == 0:
                ax.set_title(f'Community {com+1}', size=28, fontweight='bold', backgroundcolor= 'black', color='white', 
                            y=1.1, pad=1)

            disp = plotting.plot_glass_brain(mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                     cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig, axes=ax)
            disp.title(f'Pipeline {p_str}', size=24, fontweight='bold')

            thresh_mean_img, threshold = glm.threshold_stats_img(mean_img, alpha=0.05, height_control='fdr',
                                                             two_sided=False)

            ax2 = fig2.add_subplot(gs2[com, int(i)])
            
            if i == 0:
                ax2.set_title(f'Community {com+1}', size=28, fontweight='bold', backgroundcolor= 'black', color='white',
                             y=1.1, pad=1)
                
            disp2 = plotting.plot_glass_brain(thresh_mean_img, display_mode = 'z', colorbar = True, annotate=False, 
                                                     cmap=nilearn_cmaps['cold_hot'], plot_abs=False, figure=fig2, axes=ax2)
            disp2.title(f'Pipeline {p_str}', size=24, fontweight='bold')

    fig.savefig(f'{repo_path}/figures/mean_maps_per_pipeline_{data_type}_communities_{contrast}.png', dpi=300) 
    fig2.savefig(f'{repo_path}/figures/mean_thresh_maps_per_pipeline_{data_type}_communities_{contrast}.png', dpi=300) 
    
def get_activated_voxels_pipelines(contrast, partition, subject, repo_path, data_type):
    communities = [[i for i in partition.keys() if partition[i]==p] for p in np.unique(list(partition.values())).tolist()]
    
    community = {}
    for k in range(len(communities)):
        community[k] = []
        for i, s in enumerate(subject):
            if i in communities[k]:
                community[k].append(s)
                
    atlas_roi = datasets.fetch_atlas_juelich('prob-2mm')
    lab =[31,32]
    masks = [image.index_img(atlas_roi.maps, lab_i) for lab_i in lab]
    mask, mask_right, mask_left = compute_unilateral_masks(masks)
    
    if contrast == 'left-foot' or contrast == 'left-hand':
        roi_mask = mask_left # Controlateral activation
    elif contrast =='right-foot' or contrast == 'right-hand':
        roi_mask = mask_right
    else:
        roi_mask = mask

    n_communities=len(np.unique(list(partition.values())))
    
    activ_voxels = []
    thresh_activ_voxels = []
    
    for com in range(n_communities):
        com_val = []
        com_roi_val = []
        for i, p in enumerate(community[com]):
            mean_img = nib.load(f'{repo_path}/figures/mean_img_{data_type}_community_{com}_con_{contrast}.nii')

            thresh_mean_img, threshold = glm.threshold_stats_img(mean_img, alpha=0.05, height_control='fdr',
                                                             two_sided=False)
            # Count activated voxels
            img_data_act = np.nan_to_num(thresh_mean_img.get_fdata()) > 0
            n_activated_voxels = img_data_act.sum()
            com_val.append(n_activated_voxels)

            # Count activated voxels in ROI 
            mask_right = resample_from_to(mask_right, mean_img)
            mask_right_data = mask_right.get_fdata() > 1e-6
            mask_thresh_img = np.nan_to_num(thresh_mean_img.get_fdata() * mask_right_data.astype('int'))
            com_roi_val.append(np.count_nonzero(mask_thresh_img))
        
        activ_voxels.append(np.mean(com_val))
        thresh_activ_voxels.append(np.mean(com_roi_val))
        
    df = pd.DataFrame({'Community': range(n_communities), 'Whole maps': activ_voxels, 'ROI': thresh_activ_voxels})
    df.to_csv(f'{repo_path}/figures/activated_voxels_{data_type}_pipelines_{contrast}.csv')