from __future__ import print_function

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

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
use_cuda = True

opt.manualSeed = random.randint(1, 10000)  # fix seed
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

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
if use_cuda:
    classifier.cuda()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# classifier.cuda()

num_batch = len(TRAIN_DATASET) / BATCH_SIZE

strtime = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))
LOG_DIR = "log"
NAME = strtime[:13]

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(LOG_DIR + '/' + NAME): os.mkdir(LOG_DIR + '/' + NAME)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


for epoch in range(epochs):
    log_string('Epoch [%d]' % (epoch))
    losses = 0
    pt_accs = 0
    total_data = 0
    for i, data in tqdm(enumerate(train_dataloader, 0)):
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = data

        points = batch_data.transpose(2, 1)[:,:3,:]  # [bs, 3, n]
        target = batch_label # [bs, n]
        if use_cuda:
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()

        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred_all = pred.view(-1, num_classes)

        target = target.view(-1, 1)[:, 0] #- 1
        # print(pred.size(), target.size())
        loss = F.nll_loss(pred_all, target.long())
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred_all.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        losses += loss.item()
        pt_accs += correct.item() / float(NUM_POINT)
        total_data += batch_data.size()[0]

        """
        bs_box_corners = []
        for bs_index in range(BATCH_SIZE):
            print(pred[bs_index].size())
            print("pred sum", torch.max(pred[bs_index].data, dim=1)[1].sum())
            _, pred_box_corners = get_bounding_box(points[bs_index], pred[bs_index].max(1)[1])
            bs_box_corners.append(pred_box_corners)


        batch_center = batch_center.detach().numpy()
        batch_hclass = batch_hclass.detach().numpy()
        batch_hres = batch_hres.detach().numpy()
        batch_sclass = batch_sclass.detach().numpy()
        batch_sres = batch_sres.detach().numpy()
        batch_rot_angle = batch_rot_angle.detach().numpy()
        h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(batch_center[0],
                                                                  batch_hclass[0], batch_hres[0],
                                                                  batch_sclass[0], batch_sres[0],
                                                                  batch_rot_angle)
        box_corners_targ = get_3d_box((l, w, h), ry, (tx, ty, tz))


        iou_3d, iou_2d = box3d_iou(pred_box_corners, box_corners_targ)
        """
    log_string('train loss: %f' % (losses/total_data))
    log_string('train accuracy: %f' % (pt_accs/total_data))
    scheduler.step()
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, class_choice, epoch))

