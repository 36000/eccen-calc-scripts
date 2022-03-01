import neuropythy as ny, nibabel as nib, os
from AFQ.data.fetch import read_mni_template
import nibabel as nib
import numpy as np
from time import time
from dipy.align.imaffine import AffineMap
from dipy.align import resample
import matplotlib.pyplot as plt

mni = read_mni_template()
nib.save(mni, "/Users/john/AFQ_data/mni_template_for_mango.nii.gz")

eccen_img = nib.load("/Users/john/AFQ_data/benson/eccen_templ.nii.gz")
eccen_img = resample(eccen_img, mni)
nib.save(eccen_img, "/Users/john/AFQ_data/benson/eccen_templ152.nii.gz")

fov_def = 3
for roi_name in ["fov", "mac", "perip"]:
    roi = np.zeros(eccen_img.get_fdata().shape)

    if roi_name == "fov":
        roi[eccen_img.get_fdata() <= fov_def] = 1
        roi[eccen_img.get_fdata() <= 0] = 0
    elif roi_name == "mac":
        roi[eccen_img.get_fdata() <= 7] = 1
        roi[eccen_img.get_fdata() <= fov_def] = 0
    elif roi_name == "perip":
        roi[eccen_img.get_fdata() > 7] = 1

    print(f"num {roi_name} vox for " + str(fov_def) + ": " + str(np.sum(roi)))

    for side in ["L", "R"]:
        side_roi = roi.copy()
        if side == "R":
            side_roi[:side_roi.shape[0]//2, :, :] = 0
        else:
            side_roi[side_roi.shape[0]//2:, :, :] = 0
        roi_img = nib.Nifti1Image(side_roi, eccen_img.affine)
        nib.save(roi_img, f"/Users/john/AFQ_data/subroi/eccen_roi/{fov_def}/{roi_name}_{side}")
