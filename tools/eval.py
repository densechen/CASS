import copy
import glob
import json
import os

import cv2
import numpy as np
import numpy.ma as ma
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm as tqdm
from torch.autograd import Variable
import argparse

import _init_paths
import utils
from datasets.dataset import get_bbox, load_obj, PoseDataset
from lib.models import CASS
from lib.transformations import (quaternion_from_matrix,
                                 quaternion_matrix)

try:
    from metrics.evaluation_metrics import EMD_CD
except:
    raise "Failed to import EMD_CD metric. Please Compile `metric` if you want ti do reconstruction evaluation. Otherwise, just command this line."

parser = argparse.ArgumentParser(description="eval CASS model")
parser.add_argument("--resume_model", type=str, default="cass_best.pth",
                    help="resume model in 'trained_models' folder.")
parser.add_argument("--dataset_dir", type=str, default="",
                    help="dataset root of nocs")
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--draw", action="store_true", default=False,
                    help="whether to draw the pointcloud image while evaluation.")
parser.add_argument("--save_dir", type=str, default="",
                    help="dictionary to save evaluation result.")
parser.add_argument("--eval", action="store_true",
                    help="whether to re-calculate result for cass")
parser.add_argument("--mode", type=str, default="cass",
                    choices=["cass", "nocs"], help="eval cass or nocs")

opt = parser.parse_args()
opt.intrinsics = np.array(
    [[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


norm = transforms.Normalize(mean=[0.51, 0.47, 0.44], std=[0.29, 0.27, 0.28])
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 322.525
cam_cy = 244.11084
cam_fx = 591.0125
cam_fy = 590.16775
cam_scale = 1000.0
num_obj = 6
img_width = 480
img_length = 640
num_points = 500
iteration = 5
bs = 1
symmetric = [0, 1, 3]
# 0 1_bottle_02876657
# 1 2_bowl_02880940
# 2 3_camera_02942699
# 3 4_can_02946921
# 4 5_laptop_03642806
# 5 6_mug_03797390

opt.num_objects = 6
opt.num_points = 500


def to_device(x):
    if opt.cuda:
        return x.cuda()
    else:
        return x.cpu()


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.casses = self.load_model()

    def load_model(self):
        cass = CASS(self.opt)
        resume_path = os.path.join(
            "trained_models", opt.resume_model)
        try:
            cass.load_state_dict(torch.load(resume_path), strict=True)
        except:
            raise FileNotFoundError(resume_path)

        return cass

    def get_model(self, cls_idx):
        return self.casses


def get_predict_scales(recd):
    abs_coord_pts = np.abs(recd)
    return 2 * np.amax(abs_coord_pts, axis=0)


def calculate_emd_cf(point_a, point_b):
    obj = torch.from_numpy(point_a).unsqueeze(dim=0)
    pre_points = torch.from_numpy(
        point_b).unsqueeze(dim=0)
    obj = to_device(obj).float()
    pre_points = to_device(pre_points).float()

    res = EMD_CD(pre_points, obj, 1, accelerated_cd=True)
    res = {k: (v.cpu().detach().item() if not isinstance(
        v, float) else v) for k, v in res.items()}

    return res["MMD-CD"], res["MMD-EMD"]


def eval_nocs(model, img, depth, masks, cls_ids, cad_model_info, cad_model_scale):
    my_result = np.zeros((len(cls_ids), 7))
    scales = np.zeros((len(cls_ids), 3))
    chamfer_dis_cass = np.zeros((len(cls_ids)))
    emd_dis_cass = np.zeros((len(cls_ids)))

    for i in range(len(cls_ids)):
        # get model
        # cls ids zeros is not BG
        cass = model.get_model(cls_ids[i] - 1)
        try:
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(
                masks, i))  # nocs mask is start from 1
            mask = mask_label * mask_depth

            rmin, rmax, cmin, cmax = get_bbox(mask)

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([cls_ids[i] - 1])  # 0 is BG

            cloud = to_device(Variable(cloud))
            choose = to_device(Variable(choose))
            img_masked = to_device(Variable(img_masked))
            index = to_device(Variable(index))

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[
                                         1], img_masked.size()[2])

            folding_encode = cass.foldingnet.encode(img_masked, cloud, choose)
            posenet_encode = cass.estimator.encode(img_masked, cloud, choose)

            pred_r, pred_t, pred_c = cass.estimator.pose(
                torch.cat([posenet_encode, folding_encode], dim=1),
                index
            )
            recd = cass.foldingnet.recon(folding_encode)

            # get pred_scales
            scale = get_predict_scales(recd[0].detach().cpu().numpy())
            scales[i] = scale
            # load model
            for ii, info in enumerate(cad_model_info):
                if cls_ids[i] == int(info["cls_id"]):
                    model_path = info["model_path"]
                    model_scale = cad_model_scale[ii]

                    cad_model = load_obj(path=os.path.join(opt.dataset_dir, model_path[:-4]+"_{}.ply".format(
                        num_points)), ori_path=os.path.join(opt.dataset_dir, model_path), num_points=num_points)
                    # change to the real size.
                    cad_model = cad_model * model_scale

                    cd, emd = calculate_emd_cf(
                        cad_model, recd.detach()[0].cpu().numpy())
                    chamfer_dis_cass[i] = cd
                    emd_dis_cass[i] = emd
                    break
            # if detected an wrong object, we set dis to 0
            else:
                emd_dis_cass[i] = 0
                chamfer_dis_cass[i] = 0

            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            if cls_ids[i] - 1 not in symmetric:
                # Do refine for non-symmetry class and this would be useful.
                for ite in range(0, iteration):
                    T = to_device(Variable(torch.from_numpy(my_t.astype(np.float32))).view(
                        1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3))
                    my_mat = quaternion_matrix(my_r)
                    R = to_device(Variable(torch.from_numpy(
                        my_mat[:3, :3].astype(np.float32))).view(1, 3, 3))
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = cass.refiner(
                        new_cloud, folding_encode, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array(
                        [my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final
            else:
                my_pred = np.append(my_r, my_t)

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
            my_result[i] = my_pred
        except:
            # else:
            print("Empty mask while eval, skip.")
            my_result[i] = np.zeros(7)
            scales[i] = np.array([0.1, 0.1, 0.1])

            emd_dis_cass[i] = 0.0
            chamfer_dis_cass[i] = 0.0
    # convert to RTs
    my_result_ret = []
    for i in range(len(cls_ids)):
        matrix = quaternion_matrix(my_result[i][:4]).astype(np.float32)
        matrix[:3, 3] = my_result[i][4:]
        my_result_ret.append(matrix)

    return my_result_ret, scales, chamfer_dis_cass, emd_dis_cass


def eval_interface(model, opt, result):
    # do dataloading object here
    # as for gt mask the value is store in last channle, but we are store in first channel
    path = result["image_path"]
    masks = np.array(cv2.imread(os.path.join(
        opt.dataset_dir, path+"_nocs_segmentation.png"))[:, :, 0])
    img = np.array(cv2.imread(os.path.join(
        opt.dataset_dir, path+"_color.png"))) / 255.0
    depth = np.array(cv2.imread(os.path.join(
        opt.dataset_dir, path+"_depth.png"), -1))

    my_result_ret, scales, chamfer_dis_cass, emd_dis_cass = eval_nocs(
        model, img, depth, masks, result["pred_class_ids"], cad_model_info=result[
            "model_information"], cad_model_scale=result["gt_scales_for_model_in_CASS"]
    )

    my_result_ret = np.array(my_result_ret)
    scales = np.array(scales)
    chamfer_dis_cass = np.array(chamfer_dis_cass)
    emd_dis_cass = np.array(emd_dis_cass)

    return my_result_ret.tolist(), scales.tolist(), chamfer_dis_cass.tolist(), emd_dis_cass.tolist()


def draw(opt, result):
    """ Load data and draw visualization results.
    """
    path = result["image_path"]
    image = cv2.imread(os.path.join(opt.dataset_dir, path+"_color.png"))

    # Load GT Models
    models_for_nocs = []
    models_for_cass = []
    for i, mf in enumerate(result["model_information"]):
        model_path = mf["model_path"]
        cad_model = load_obj(path=os.path.join(opt.dataset_dir, model_path[:-4]+"_{}.ply".format(
            num_points)), ori_path=os.path.join(opt.dataset_dir, model_path), num_points=num_points)

        # As for nocs, the model normalized is gt points.
        models_for_nocs.append(copy.deepcopy(cad_model))

        # As for cass, the model normalized should multiply scale to get the real size.
        models_for_cass.append(copy.deepcopy(
            cad_model * result["gt_scales_for_model_in_CASS"][i]))

    # Get the correct RTs for Class_ids. If the target is missing we will return np.eye(). If multi-target is matched, we only keep the first.
    RTs_cass = []
    RTs_nocs = []
    misses = []
    for i, cls in enumerate(result["gt_class_ids"]):
        idx = result["pred_class_ids"] == cls
        rts_nocs = result["pred_RTs"][idx]

        rts_cass = result["pred_RTs_cass"][idx]

        miss = False
        if len(rts_cass) <= 0 or len(rts_nocs) <= 0:
            rts_cass = np.eye(4)
            rts_nocs = np.eye(4)
            miss = True
        elif len(rts_cass) > 1 or len(rts_nocs) > 1:
            rts_cass = rts_cass[0]
            rts_nocs = rts_nocs[0]
        misses.append(miss)
        RTs_nocs.append(rts_nocs)
        RTs_cass.append(rts_cass)

    (h, w) = image.shape[:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    utils.draw(rotated, RTs_cass, models_for_cass, class_ids=result["gt_class_ids"], misses=misses, intrinsics=opt.intrinsics, save_path=os.path.join(
        opt.save_dir, "vis", "_".join(path.split("/"))+"_cass.png"))
    utils.draw(image, RTs_nocs, models_for_nocs, class_ids=result["gt_class_ids"], misses=misses, intrinsics=opt.intrinsics, save_path=os.path.join(
        opt.save_dir, "vis", "_".join(path.split("/"))+"_nocs.png"))


if __name__ == "__main__":
    opt.class_names = PoseDataset.get_class_names()

    eval_dir = os.path.join(opt.save_dir, "eval_{}".format(opt.mode))
    os.makedirs(eval_dir, exist_ok=True)

    if opt.mode == "cass":

        if opt.eval:
            model = to_device(Model(opt)).eval()

        result_json_list = glob.glob(
            os.path.join(opt.save_dir, "gt", "*.json"))
        result_json_list = sorted(result_json_list)

        final_results = []
        for filename in tqdm.tqdm(result_json_list, desc="loading"):

            if opt.eval:
                with open(filename, "r") as f:
                    result = json.load(f)

                pred_RTs_cass, pred_scales_cass, chamfer_dis_cass, emd_dis_cass = eval_interface(
                    model, opt, result)

                result["pred_RTs_cass"] = pred_RTs_cass
                result["pred_scales_cass"] = pred_scales_cass

                result["chamfer_dis_cass"] = chamfer_dis_cass
                result["emd_dis_cass"] = emd_dis_cass

                with open(os.path.join(eval_dir, os.path.basename(filename)), "w") as f:
                    json.dump(result, f, indent=4)
            else:
                with open(os.path.join(eval_dir, os.path.basename(filename)), "r") as f:
                    result = json.load(f)

            gt_class_ids = []
            gt_scales_for_CASS = []
            for m in result["model_information"]:
                gt_class_ids.append(int(m["cls_id"]))
                gt_scales_for_CASS.append(m["gt_scales_for_CASS"])
            result["gt_class_ids"] = gt_class_ids
            result["gt_handle_visibility"] = [1] * len(gt_class_ids)
            result["gt_scales_for_CASS"] = gt_scales_for_CASS

            # convert all label information to np.array if possible
            r = {}
            for k, v in result.items():
                if isinstance(v, (list, tuple)):
                    r[k] = np.array(v)
                else:
                    r[k] = v
            final_results.append(r)

        if opt.draw:
            os.makedirs(os.path.join(opt.save_dir, "vis"), exist_ok=True)
            for r in tqdm.tqdm(final_results, desc="draw"):
                draw(opt, r)

        synset_names = ["BG"] + opt.class_names

        # eval
        eval_results = []
        for i in final_results:
            i["pred_scales"] = i["pred_scales_cass"]
            i["pred_RTs"] = i["pred_RTs_cass"]
            i["pred_class_ids"] = i["pred_class_ids"]
            i["gt_scales"] = i["gt_scales_for_CASS"]
            i["gt_RTs"] = i["gt_RTs_for_CASS"]
            eval_results.append(i)
        aps = utils.compute_degree_cm_mAP(
            eval_results, synset_names, eval_dir,
            degree_thresholds=range(0, 61, 1),
            shift_thresholds=np.linspace(0, 1, 31)*15,
            iou_3d_thresholds=np.linspace(0, 1, 101),
            iou_pose_thres=0.1,
            use_matches_for_pose=True, eval_recon=True
        )
    elif opt.mode == "nocs":
        result_json_list = glob.glob(
            os.path.join(opt.save_dir, "gt", "*.json"))
        result_json_list = sorted(result_json_list)

        final_results = []
        for filename in tqdm.tqdm(result_json_list, desc="loading"):
            with open(os.path.join(filename), "r") as f:
                result = json.load(f)

            # convert all label information to np.array if possible
            r = {}
            for k, v in result.items():
                if isinstance(v, (list, tuple)):
                    r[k] = np.array(v)
                else:
                    r[k] = v
            final_results.append(r)

        synset_names = ["BG"] + opt.class_names

        aps = utils.compute_degree_cm_mAP(
            final_results, synset_names, eval_dir,
            degree_thresholds=range(0, 61, 1),
            shift_thresholds=np.linspace(0, 1, 31)*15,
            iou_3d_thresholds=np.linspace(0, 1, 101),
            iou_pose_thres=0.1,
            use_matches_for_pose=True, eval_recon=False
        )
