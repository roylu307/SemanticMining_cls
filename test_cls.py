from data_utils.ModelNetDataLoader import ModelNetDataLoader
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F
import argparse
import numpy as np
import os
import re
import glob
import torch
import logging
from tqdm import tqdm
import sys
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='smnet_9layer', help='Experiment root')
    parser.add_argument('--model', type=str, default='best_model', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=10, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--num_repeat', type=int, default=50, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--from_epoch', type=int, default=100, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--eval_all', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size(0)
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

def test(classifier, loader, num_class=40, vote_num=1):
    pc_rand_scale = PointcloudScale()
    total_correct = 0
    total_seen = 0
    classifier = classifier.eval()
    with torch.no_grad():
        for j, data in tqdm(enumerate(loader), total=len(loader)):
            
            points, target = data
            target = target[:, 0]

            # points[:,:, 0:3] = shift_point_cloud(points[:,:, 0:3], shift_range=100)
            points[:,:, 0:3] = rotate_point_cloud(points[:,:, 0:3], angle_range=1) # angle_range * pi
            # points[:,:, 0:3] = scale_point_cloud(points[:,:, 0:3], scale_low=5.0, scale_high=5.0)

            points, target = points.cuda(), target.cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200*(args.num_point//1024))

            vote_pool = 0
            for v in range(vote_num):
                new_fps_idx = fps_idx[:, np.random.choice(1200*(args.num_point//1024), args.num_point, False)]
                new_points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()

                if v > 0:
                    new_points.data = pc_rand_scale(new_points.data)

                pred = classifier(new_points)
                pred = F.softmax(pred, dim = 1)
                vote_pool += pred
            pred = vote_pool/vote_num
            _, pred_choice = torch.max(pred.data,-1)
            
            correct = pred_choice.eq(target.long().data).cpu().sum()
            total_correct += correct.item()
            total_seen += float(points.size()[0])

        oa_acc = total_correct/total_seen
        return oa_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    model_dir = experiment_dir + '/checkpoints/'
    sys.path.append(experiment_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_name = '%s/eval_%s.txt' % (experiment_dir, 'all') if args.eval_all else '%s/eval_%s.txt' % (experiment_dir, args.model)
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40/modelnet40_normal_resampled/'
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point*2, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 40
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    print(experiment_dir)

    classifier = MODEL.DensePoint(num_classes = 40, input_channels = 0, use_xyz = True).cuda()

    

    if not args.eval_all:
        global_acc = 0
        checkpoint = torch.load(model_dir + args.model + '.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Current Epoch: %s' % args.model)
        for i in range(args.num_repeat):
            acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
            
            if acc > global_acc:
                global_acc = acc
            log_string('Repeat (%3d/%3d) \t Acc: %0.6f' % (i+1, args.num_repeat, acc))
            log_string('Best voting acc: %0.6f' % (global_acc))


    if args.eval_all:
        global_acc = 0
        ckpt_list = glob.glob(os.path.join(model_dir, 'model_*.pth'))
        ckpt_list.sort(key=os.path.getmtime)
        for j, cur_ckpt in enumerate(ckpt_list):
            num_list = re.findall('model_(.*).pth', cur_ckpt)
            epoch_id=num_list[-1]
            if int(float(epoch_id)) > args.from_epoch:
                log_string('Current Epoch: model_%s (%d/%d)' % (epoch_id, j, len(ckpt_list)))
                checkpoint = torch.load(cur_ckpt)
                classifier.load_state_dict(checkpoint['model_state_dict'])

                
                for i in range(args.num_repeat):
                    acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes)
                    
                    if acc > global_acc:
                        global_acc = acc
                    log_string('Epoch: %s \t Repeat %3d \t Acc: %0.6f' % (epoch_id, i + 1, acc))
                    log_string('Best voting acc: %0.6f' % (global_acc))

def scale_point_cloud(pc, scale_low=0.1, scale_high=10):
    bsize = pc.size(0)
    for i in range(bsize):
        xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
        
        pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float())
        
    return pc

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_point_cloud(batch_data, angle_range=0.25):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform(low=0, high=angle_range) * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return torch.from_numpy(rotated_data)

if __name__ == '__main__':
    args = parse_args()
    main(args)
