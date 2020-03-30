import os

import cv2
import numpy as np
import numpy.ma as ma
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import datasets.utils as dutils
from lib.transformations import quaternion_from_matrix


def align(class_ids, masks, coords, depth, intr):
    num_instances = len(class_ids)
    RTs = np.zeros((num_instances, 4, 4), dtype=np.float32)
    scales = np.ones((num_instances, 3), dtype=np.float32)

    for i in range(num_instances):
        mask = ma.getmaskarray(ma.masked_equal(masks, class_ids[i]))
        if np.sum(mask) < 50:
            RTs[i] = np.eye(4)
            continue

        pts, idxs = dutils.backproject(depth, intr, mask)
        pts = pts / 1000.0
        if len(pts) < 50:
            RTs[i] = np.eye(4)
            continue
        coord_pts = coords[idxs[0], idxs[1], :] - 0.5

        scale, rotation, trans, _ = dutils.estimateSimilarityTranform(
            coord_pts, pts)
        if rotation is None or trans is None or np.any(np.isnan(rotation)) or np.any(np.isnan(trans))\
                or np.any(np.isinf(trans)) or np.any(np.isinf(rotation)):
            RTs[i] = np.eye(4)
            continue

        aligned_RT = np.eye(4)
        aligned_RT[:3, :3] = rotation.T

        aligned_RT[:3, 3] = trans
        aligned_RT[3, 3] = 1

        RTs[i, :, :] = aligned_RT
        scales[i] = scale

    return RTs, scales


def load_obj(path, ori_path, num_points):
    if os.path.isfile(path):
        return dutils.load_obj(path)
    else:
        vertex = dutils.sample_obj(ori_path, num_points, True)
        dutils.save_obj(vertex, path[:-3]+"ply")
        return np.asarray(vertex)


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, root):
        if mode == 'train':
            self.path = 'datasets/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        with open(self.path, "r") as input_file:
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line.replace("\n", "")
                if input_line.startswith("real"):
                    self.real.append(input_line)
                else:
                    self.syn.append(input_line)
                self.list.append(input_line)

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)
        # real
        self.cam_cx_1 = 322.525
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.0125
        self.cam_fy_1 = 590.16775
        # syn
        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.minimum_num_pt = 50

        self.norm = transforms.Normalize(
            mean=[0.51, 0.47, 0.44], std=[0.29, 0.27, 0.28])
        self.symmetry_obj_idx = [0, 1, 3]

        self.class_names = PoseDataset.get_class_names()
        print(len(self.list))

    @staticmethod
    def get_class_names():
        class_names = []
        with open("datasets/dataset_config/classes.txt", "r") as f:
            class_names = ["_".join(line.split("_")[1:2]) for line in f]
        class_names = [c.replace("\n", "") for c in class_names]

        return class_names

    def __getitem__(self, index):
        try:
            img = np.array(cv2.imread(
                '{0}/{1}_color.png'.format(self.root, self.list[index]))) / 255.
            depth = np.array(cv2.imread(
                '{0}/{1}_depth.png'.format(self.root, self.list[index]), -1))
            if len(depth.shape) == 3:
                depth = np.uint16(depth[:, :, 1] * 256) + \
                    np.uint16(depth[:, :, 2])
            label = np.array(cv2.imread(
                '{0}/{1}_mask.png'.format(self.root, self.list[index]))[:, :, 2])

            meta = dict()
            with open("{0}/{1}_meta.txt".format(self.root, self.list[index]), "r") as f:
                for line in f:
                    line = line.replace("\n", "")
                    line = line.split(" ")
                    if int(line[1]) == 0:  # mask out background
                        continue
                    d = {"cls_id": line[1], "inst_name": line[2]}
                    if "real_train" in self.list[index]:
                        d["inst_dir"] = os.path.join(self.root, "obj_models", "real_train",
                                                     line[2]+"_{}.ply".format(self.num_pt))
                        d["ori_inst_dir"] = os.path.join(self.root,
                                                         "obj_models", "real_train", line[2]+".obj")
                    elif "real_test" in self.list[index]:
                        d["inst_dir"] = os.path.join(self.root, "obj_models", "real_test",
                                                     line[2]+"_{}.ply".format(self.num_pt))
                        d["ori_inst_dir"] = os.path.join(
                            self.root, "obj_models", "real_test", line[2]+".obj")
                    else:
                        d["inst_dir"] = os.path.join(self.root, "obj_models", "train",
                                                     *line[2:], "model_{}.ply".format(self.num_pt))
                        d["ori_inst_dir"] = os.path.join(self.root, "obj_models", "train",
                                                         *line[2:], "model.obj")
                    meta[int(line[0])] = d

            if not self.list[index].startswith("real"):
                cam_cx = self.cam_cx_2
                cam_cy = self.cam_cy_2
                cam_fx = self.cam_fx_2
                cam_fy = self.cam_fy_2
            else:
                cam_cx = self.cam_cx_1
                cam_cy = self.cam_cy_1
                cam_fx = self.cam_fx_1
                cam_fy = self.cam_fy_1

            obj = list(meta.keys())
            iidx = np.arange(len(obj))
            np.random.shuffle(iidx)
            for idx in iidx:
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
                mask = mask_label * mask_depth
                if len(mask.nonzero()[0]) > self.minimum_num_pt:
                    break
            else:
                print("Can't find any valid training object in {}".format(
                    self.list[index]))
                raise ValueError

            # A method to load target_r and target_t
            if os.path.isfile("{}/gts/{}_poses.txt".format(self.root, self.list[index])) and os.path.isfile("{}/gts/{}_scales.txt".format(self.root, self.list[index])):
                meta["poses"] = np.loadtxt(
                    "{}/gts/{}_poses.txt".format(self.root, self.list[index])).reshape(-1, 4, 4)
                meta["scales"] = np.loadtxt(
                    "{}/gts/{}_scales.txt".format(self.root, self.list[index])).reshape(-1, 3)
            else:
                coord = cv2.imread(
                    '{0}/{1}_coord.png'.format(self.root, self.list[index]))[:, :, :3][:, :, (2, 1, 0)]
                coord = np.array(coord, dtype=np.float32) / 255.
                coord[:, :, 2] = 1.0 - coord[:, :, 2]
                intr = np.array(
                    [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0., 0., 1.]])
                poses, scales = align(obj, label, coord, depth, intr)
                os.makedirs(os.path.dirname(
                    "{}/gts/{}_poses.txt".format(self.root, self.list[index])), exist_ok=True)
                np.savetxt("{}/gts/{}_poses.txt".format(self.root, self.list[index]),
                           poses.reshape(-1, 4))
                np.savetxt("{}/gts/{}_scales.txt".format(self.root,
                                                         self.list[index]), scales.reshape(-1, 3))
                meta["poses"] = poses
                meta["scales"] = scales
            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            img_masked = np.transpose(img, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            target_r = meta['poses'][idx][:3, 0:3]
            target_t = np.array([meta['poses'][idx][:3, 3:4].flatten()])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten(
            )[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = 1000.0
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

            model_points = load_obj(
                path=meta[obj[idx]]["inst_dir"],
                ori_path=meta[obj[idx]]["ori_inst_dir"], num_points=self.num_pt)

            model_points = model_points * meta["scales"][idx]

            target = np.dot(model_points, target_r.T)
            target = np.add(target, target_t)
            matrix = np.eye(4)
            matrix[:3, :3] = target_r
            quat = quaternion_from_matrix(matrix)

            return torch.from_numpy(cloud.astype(np.float32)), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)), \
                torch.from_numpy(model_points.astype(np.float32)), \
                torch.LongTensor([int(meta[obj[idx]]["cls_id"])-1]), \
                torch.from_numpy(quat.astype(np.float32)), \
                torch.from_numpy(target_t.astype(np.float32))
        except:
            return self.__getitem__(index//2)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt


border_list = [-1, 40, 80, 120, 160, 200, 240, 280,
               320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
