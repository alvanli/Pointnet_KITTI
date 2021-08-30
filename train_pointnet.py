from __future__ import print_function

import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from frustum_utils import get_bounding_box, from_prediction_to_label_format, get_3d_box, box3d_iou, FrustumDataset
import time
import open3d as o3d


manual_seed = 42
class_choice = "Car"
epochs = 10
BATCH_SIZE = 32
workers = 4
NUM_POINT = 1024
train_sets = "train"
val_sets = "val"
objtype = "carpedcyc"
num_classes = 2
feature_transform = True
use_cuda = True

print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# TRAIN_DATASET = FrustumDataset(npoints=NUM_POINT, split=train_sets,
#     rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
#     overwritten_data_path='/home/aldec/Data/WAT/kitti/kitti/frustum_'+objtype+'_'+train_sets+'.pickle')
# TEST_DATASET = FrustumDataset(npoints=NUM_POINT, split=val_sets,
#     rotate_to_center=True, one_hot=True,
#     overwritten_data_path='/home/aldec/Data/WAT/kitti/kitti/frustum_'+objtype+'_'+val_sets+'.pickle')

TRAIN_DATASET = FrustumDataset(npoints=NUM_POINT, split=train_sets,
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True,
    overwritten_data_path='/mnt/wato-drive/KITTI/pickle/frustum_'+objtype+'_'+train_sets+'.pickle')
TEST_DATASET = FrustumDataset(npoints=NUM_POINT, split=val_sets,
    rotate_to_center=True, one_hot=True,
    overwritten_data_path='/mnt/wato-drive/KITTI/pickle/frustum_'+objtype+'_'+val_sets+'.pickle')

train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,\
                                num_workers=8, pin_memory=True)
test_dataloader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,\
                                num_workers=8, pin_memory=True)

print(len(train_dataloader), len(test_dataloader))


blue = lambda x: '\033[94m' + x + '\033[0m'

Pointnet = PointNetDenseCls(k=num_classes, feature_transform=feature_transform)
if use_cuda:
    Pointnet.cuda()

# if not use_cuda:
Pointnet.load_state_dict(torch.load("log/2021-08-29-19/seg_model_Car_5.pth", map_location= torch.device('cpu')))

optimizer = optim.Adam(Pointnet.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# classifier.cuda()

num_batch = len(TRAIN_DATASET) / BATCH_SIZE

strtime = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
LOG_DIR = "log"
NAME = strtime[:13]

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(LOG_DIR + '/' + NAME): os.mkdir(LOG_DIR + '/' + NAME)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def one_epoch(dataloader, classifier, best_iou3d, is_train=True):
    all_losses = 0
    all_pt_acc = 0
    all_iou3d = 0
    all_iou_acc = 0

    total_data = 0
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = data

        points = batch_data.transpose(2, 1)[:, :3, :]  # [bs, 3, n]
        target = batch_label  # [bs, n]
        if use_cuda:
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()

        if is_train:
            classifier = classifier.train()
        else:
            classifier = classifier.eval()

        pred, trans, trans_feat = classifier(points)
        pred_all = pred.view(-1, num_classes)

        target = target.view(-1, 1)[:, 0]  # - 1
        # print(pred.size(), target.size())
        loss = F.nll_loss(pred_all, target.long())
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        if is_train:
            loss.backward()
            optimizer.step()

        pred_choice = pred_all.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        all_losses += loss.item()
        all_pt_acc += correct.item() / float(NUM_POINT)
        total_data += batch_data.size()[0]

        bs_box_corners = []
        for bs_index in range(batch_data.size()[0]):
            _, pred_box_corners = get_bounding_box(points[bs_index], pred[bs_index].max(1)[1])
            re_id = [1, 0, 4, 5, 2, 3, 7, 6]
            pred_box_corners = pred_box_corners[re_id]
            bs_box_corners.append(pred_box_corners)

        batch_center = batch_center.detach().numpy()
        batch_hclass = batch_hclass.detach().numpy()
        batch_hres = batch_hres.detach().numpy()
        batch_sclass = batch_sclass.detach().numpy()
        batch_sres = batch_sres.detach().numpy()
        batch_rot_angle = batch_rot_angle.detach().numpy()

        for bs_index in range(BATCH_SIZE):
            h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(batch_center[bs_index],
                                                                      batch_hclass[bs_index], batch_hres[bs_index],
                                                                      batch_sclass[bs_index], batch_sres[bs_index],
                                                                      batch_rot_angle[bs_index])
            box_corners_targ = get_3d_box((l, w, h), ry, (tx, ty, tz))

            iou_3d, _ = box3d_iou(bs_box_corners[bs_index], box_corners_targ)
            if iou_3d > 0.7:
                all_iou_acc += 1
            all_iou3d += iou_3d

            # xyz = batch_data.detach().numpy()[bs_index, :, :3]
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            #
            # lines = [[0, 1], [1, 2], [1, 5], [0, 3], [0, 4], [2, 3], [2, 6],
            #          [3, 7], [4, 5], [5, 6], [4, 7], [6, 7]]
            # colors = [[0, 1, 0] for _ in range(len(lines))]
            # line_set = o3d.geometry.LineSet()
            # line_set.points = o3d.utility.Vector3dVector(box_corners_targ)
            # line_set.lines = o3d.utility.Vector2iVector(lines)
            # line_set.colors = o3d.utility.Vector3dVector(colors)
            #
            # colors = [[1, 0, 0] for _ in range(len(lines))]
            # pred_line_set = o3d.geometry.LineSet()
            # pred_line_set.points = o3d.utility.Vector3dVector(bs_box_corners[bs_index])
            # pred_line_set.lines = o3d.utility.Vector2iVector(lines)
            # pred_line_set.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd, line_set, pred_line_set])
            # input()

    word = "train" if is_train else "test"
    each_iou3d = all_iou3d / total_data
    log_string('%s loss: %f' % (word, all_losses / total_data))
    log_string('%s accuracy: %f' % (word,all_pt_acc / total_data))
    log_string('%s iou3d: %f' % (word, each_iou3d))
    log_string('%s  iou acc: %f' % (word, all_iou_acc / total_data))

    if is_train:
        scheduler.step()
        if each_iou3d > best_iou3d:
            torch.save(classifier.state_dict(), '%s/%s/seg_model_%s_%d.pth' % (LOG_DIR, NAME, class_choice, epoch))
            return each_iou3d
    return 0


best_iou3d_global = 0
for epoch in range(epochs):
    log_string('Epoch [%d]' % (epoch))
    return_iou3d = one_epoch(train_dataloader, Pointnet, best_iou3d_global, is_train=True)
    if return_iou3d:
        best_iou3d_global = return_iou3d
    one_epoch(test_dataloader, Pointnet, best_iou3d_global, is_train=False)






