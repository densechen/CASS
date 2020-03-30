# CASS:Learning Canonical Shape Space for Category-Level 6D Object Pose and Size Estimation

## Evaluation Steps
1.  Install the following requirements:

```
    open3d==0.8.0.0
    opencv-python==4.1.1.26
    torch==1.2.0
    torchvision==0.4.0
    tqdm==4.32.1
    trimesh==3.2.20
```

1.  Compile "./metrics" for **re-evaluating** reconstructed models. You can skip this step and delete line 25-28 in ./tools/eval.py, if you have downloaded our results in next step.

2.  Download predicted masks and pretrained models.
    
    You can download our pretrained models, results and segmentation masks of real test dataset in [NOCS](https://github.com/hughw19/NOCS_CVPR2019) from [Google Driver](https://drive.google.com/drive/folders/1yvVpvB_0YuqNAaeOzE5YfO5dvwDaoz_n). 
    
    If you want to **re-calculate** CASS's results, please download the NOCS [real test dataset](http://download.cs.stanford.edu/orion/nocs/real_test.zip) and [3d models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip).

3.  Evaluate CASS and NOCS

    1.  Unzip predicted results, and specified `--save_dir` in eval.sh. You will get evaluation results of CASS and NOCS at the same time.
    2.  If you want to recalculate CASS's results, please place segmentation mask of NOCS, which is contained in the Google Driver, to the real-test dataset folder along with their color images. Refer to 1-2 line in ./eval.sh about how to start the evaluation.

## Acknowledgement

We have referred to part of the code from [NOCS_CVPR2019](https://github.com/hughw19/NOCS_CVPR2019), [FoldingNet](https://github.com/jtpils/FoldingNet), [DenseFusion](https://github.com/j96w/DenseFusion), [Open3D](https://github.com/intel-isl/Open3D) and [PointFlow](https://github.com/stevenygd/PointFlow/tree/master).
