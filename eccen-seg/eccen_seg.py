from dipy.io.streamline import save_tractogram
from AFQ.segmentation import Segmentation
from AFQ.definitions.mapping import ConformedFnirtMapping
from AFQ.registration import read_mapping
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import nibabel as nib
import numpy as np
import sys

from fsl.data.image import Image
from fsl.transform.fnirt import readFnirt
from fsl.utils.path import PathError

screen = sys.argv[1]
batch = sys.argv[2]
SUB = sys.argv[3]
roi_choice = sys.argv[4]

fa_path = f"data/jk232/optic_bids_{screen}_{batch}/derivatives/afq/sub-{SUB}/sub-{SUB}_dwi_model-DKI_FA.nii.gz"
fa_img = nib.load(fa_path)
if "gcalc" in roi_choice:
    reg_template = nib.load("data/jk232/AFQ_data/subroi/G_Calc_1_R.nii")
else:
    reg_template = nib.load(f"data/jk232/AFQ_data/subroi/eccen_roi/1/fov_R.nii")
try:
    nearest_warp = f"data/jk232/optic_bids_{screen}_{batch}/derivatives/TBSS/sub-{SUB}/sub-{SUB}_dwi_MNI_to_dti_FA_warp.nii.gz"
    nearest_space = f"data/jk232/optic_bids_{screen}_{batch}/derivatives/TBSS/sub-{SUB}/sub-{SUB}_dwi_dti_FA_to_MNI.nii.gz"
    subj = Image(fa_path)
    their_templ = Image(nearest_space)
    warp = readFnirt(nearest_warp, their_templ, subj)
    mapping = ConformedFnirtMapping(warp, reg_template.affine)
except PathError:
    reg_prealign = np.load(
        f"data/jk232/optic_bids_{screen}_{batch}/derivatives/afq/sub-{SUB}/sub-{SUB}_dwi_prealign_from-DWI_to-MNI_xfm.npy")
    reg_prealign_inv = np.linalg.inv(reg_prealign)
    mapping = read_mapping(
        f"data/jk232/optic_bids_{screen}_{batch}/derivatives/afq/sub-{SUB}/sub-{SUB}_dwi_mapping_from-DWI_to_MNI_xfm.nii.gz",
        fa_path,
        reg_template,
        prealign=reg_prealign_inv)

def get_rois(side):
    fov = nib.load(f"data/jk232/AFQ_data/subroi/eccen_roi/3/fov_{side}.nii")
    mac = nib.load(f"data/jk232/AFQ_data/subroi/eccen_roi/3/mac_{side}.nii")
    perip = nib.load(f"data/jk232/AFQ_data/subroi/eccen_roi/3/perip_{side}.nii")
    return fov, mac, perip

num_dict = {"one": 1, "two": 2, "three": 3}

startpoint = nib.Nifti1Image(np.ones(fa_img.get_fdata().shape), fa_img.affine)
for side in ["R", "L"]:
    bundle_dict = {}
    if roi_choice == "gcalc":
        for i in range(2):
            if i == 0:
                this_roi1 = nib.load(f"data/jk232/AFQ_data/subroi/G_Calc_1_{side}.nii")
                this_roi2 = nib.load(f"data/jk232/AFQ_data/subroi/G_Calc_2_{side}.nii")
                this_roi = nib.Nifti1Image(np.logical_or(this_roi1.get_fdata().astype(bool), this_roi2.get_fdata().astype(bool)).astype(np.float32), this_roi1.affine)
            else:
                this_roi = nib.load(f"data/jk232/AFQ_data/subroi/G_Calc_3_{side}.nii")
            bundle_dict[f"gcalc{i}_{side}"] = {
                'include': [this_roi, this_roi],
                "start": startpoint, "end": this_roi,
                'cross_midline': False, 'prob_map': this_roi.get_fdata(), 'uid': i}
    elif roi_choice in ["three"]:
        number = num_dict[roi_choice]
        for roi_name, roi, i in zip(["fov", "mac", "perip"], get_rois(number, side), [0, 1, 2]):
            bundle_dict[f"{roi_name}_{number}_{side}"] = {
                'include': [roi, roi],
                "start": startpoint, "end": roi,
                'cross_midline': False, 'uid': i}
    
    trk = nib.streamlines.load(f"data/jk232/optic_bids_{screen}_{batch}/derivatives/afq/sub-{SUB}/clean_bundles/sub-{SUB}_dwi_space-RASMM_model-DTI_desc-det-AFQ-{side}_OR_tractography.trk")
    if len(trk.streamlines) == 0:
        for key in bundle_dict.keys():
           save_tractogram(StatefulTractogram(trk.streamlines, fa_img, Space.RASMM), f"data/jk232/optic_bids_{screen}_{batch}/derivatives/gcalc/sub-{SUB}/sls/{key}.trk", bbox_valid_check=False)
        continue
    tg = StatefulTractogram(trk.streamlines, fa_img, Space.RASMM)
    tg.to_vox()
    seg = Segmentation(
        save_intermediates=f"data/jk232/optic_bids_{screen}_{batch}/derivatives/gcalc/sub-{SUB}/gcalc_inters",
        roi_dist_tie_break=True)
    seg.img = fa_img
    fg = seg.segment(bundle_dict, tg, mapping=mapping, img_affine=fa_img.affine, reg_template=reg_template)
    for key, val in fg.items():
        print(len(fg[key]))
        save_tractogram(
            fg[key],
            f"data/jk232/optic_bids_{screen}_{batch}/derivatives/gcalc/sub-{SUB}/sls/{key}.trk",
            bbox_valid_check=False)

