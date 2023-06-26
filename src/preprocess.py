from glob import glob
from nilearn.image import resample_img, resample_to_img
from nilearn import datasets, plotting, masking
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os.path as op
import os

def preprocessing(data_dir, output_dir, resolution):
    '''
    Preprocess all maps that are stored in the 'original' repository of the data_dir. 
    Store these maps in subdirectories of the data_dir corresponding to the preprocessing step applied.

    Parameters:
        - data_dir, str: path to directory where 'original' directory containing all original images is stored
        - output_dir, str: path to directory where 'preprocessed' directory is stored and where all preprocessed images will be stored.

    '''
    # Get image list to preprocess
    img_list = sorted(glob(f'{data_dir}/group-*_*_*_tstat.nii'))
        
    # Create dirs to save images
    if not op.isdir(op.join(output_dir, f'resampled_mni_masked_res_{resolution}')):
        os.mkdir(op.join(output_dir, f'resampled_mni_masked_res_{resolution}'))

    if not op.isdir(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}')):
        os.mkdir(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}'))
    
    for idx, img in enumerate(img_list):

        print('Image', img)

        if not os.path.exists(op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}', op.basename(img))):

            nib_img = nib.load(img)
            img_data = nib_img.get_fdata()
            mask_data = np.isnan(nib_img.get_fdata()).astype('float')
            img_data = np.nan_to_num(img_data)
            img_affine = nib_img.affine

            nib_img = nib.Nifti1Image(img_data, img_affine)
            nib_mask = nib.Nifti1Image(mask_data, img_affine)

            print('Original shape of image ', idx+1, ':',  nib_img.shape)

            try:

                print("Resampling image {0} of {1}...".format(idx + 1, len(img_list)))
                mni_res_img = resample_to_img(nib_img, datasets.load_mni152_template(resolution=resolution))

                print("Masking image {0} of {1}...".format(idx + 1, len(img_list)))
                mni_res_mask = resample_to_img(nib_mask, datasets.load_mni152_template(resolution=resolution))         
                mni_res_mask_data = np.array(mni_res_mask.get_fdata() < 0.5).astype('float')                                          
                mni_res_img_data = mni_res_img.get_fdata()
                mni_res_masked_img_data = mni_res_img_data * mni_res_mask_data
                mni_res_masked_img_data = mni_res_masked_img_data[0:48,0:56,0:48]
                mni_res_masked_img = nib.Nifti1Image(mni_res_masked_img_data, mni_res_img.affine)
                #nib.save(mni_res_masked_img, op.join(output_dir,f'resampled_mni_masked_res_{resolution}', op.basename(img))) # Save original image resampled and masked

                print('Min-Max normalizing masked image', idx)
                res_masked_norm_img_data = mni_res_masked_img_data.copy().astype(float)
                res_masked_norm_img_data = np.nan_to_num(res_masked_norm_img_data)
                res_masked_norm_img_data *= 1.0/np.abs(res_masked_norm_img_data).max()
                res_masked_norm_img_data = res_masked_norm_img_data[0:48,0:56,0:48]
                res_masked_norm_img = nib.Nifti1Image(res_masked_norm_img_data, mni_res_img.affine)
                nib.save(res_masked_norm_img, op.join(output_dir, f'resampled_mni_masked_normalized_res_{resolution}', op.basename(img))) # Save original image resampled, masked and normalized

                print(f"Image {idx} : DONE.")

            except Exception as e:
                
                print("Failed!")
                print(e)
                continue
        else:
            print('Image already preprocessed.')

def main():
    data_dir = '/srv/tempdd/egermani/hcp_many_pipelines'
    out_dir = '/srv/tempdd/egermani/hcp_many_pipelines_preprocessed'
    resolution = 4
    preprocessing(data_dir, out_dir, resolution)

if __name__ == '__main__':
    main()
           