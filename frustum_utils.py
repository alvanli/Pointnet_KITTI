import numpy as np
import cv2
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512
NUM_HEADING_BIN = 12

g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs


def get_bounding_box(points, target):
    pts_np, targ_np = points.data.numpy(), target.data.numpy()
    obj = pts_np[:, :2, targ_np[0, :] == 1]
    rect = cv2.minAreaRect(obj[0].transpose())
    box = cv2.boxPoints(rect)
    box = np.concatenate((box, box), axis=0)

    centre, width, angle = rect
    max_z = np.max(pts_np[:, 2, targ_np[0, :] == 1])
    min_z = np.min(pts_np[:, 2, targ_np[0, :] == 1])
    a = np.array([np.append(np.ones((4)) * max_z, np.ones((4)) * min_z)]).transpose()
    box = np.append(box, a, axis=1)
    height_z = np.ptp(pts_np[:, 2, targ_np[0, :] == 1])
    centre_z = height_z / 2
    return [(centre[0], centre[1], centre_z), (width[0], width[1], height_z), angle], box


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=object)

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    # ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join(ROOT_DIR,
                                                 'kitti/frustum_carpedcyc_%s.pickle' % (split))

        self.from_rgb_detection = from_rgb_detection
        if from_rgb_detection:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            with open(overwritten_data_path, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                # frustum_angle is clockwise angle from positive x-axis
                self.frustum_angle_list = pickle.load(fp)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)
        # np.pi/2.0 + self.frustum_angle_list [index]float,[-pi/2,pi/2]

        # Compute one hot vector
        if self.one_hot:  # True
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:  # True
            point_set = self.get_center_view_point_set(index)  # (n,4) #pts after Frustum rotation
        else:
            point_set = self.input_list[index]

        # ipdb.set_trace()
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.from_rgb_detection:
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]  # (1024,),array([0., 1., 0., ..., 1., 1., 1.])

        # Get center point of 3D box
        if self.rotate_to_center:  # True
            box3d_center = self.get_center_view_box3d_center(index)  # array([ 0.07968819,  0.39      , 46.06915834])
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:  # True
            heading_angle = self.heading_list[index] - rot_angle  # -1.6480684951683866 #alpha
        else:
            heading_angle = self.heading_list[index]  # rotation_y

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])  # 5, array([0.25717603, 0.00293633, 0.12301873])
        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)

        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))
