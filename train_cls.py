from data_utils.ModelNetDataLoader import ModelNetDataLoader
from pointnet2_ops import pointnet2_utils
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='smnet_9layer', help='model name [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str, default='smnet_cls', help='experiment root')
    parser.add_argument('--use_ckpt', action='store_true', default=False, help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=500, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    # DATA
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--num_worker', type=int, default=16, help='number of dataloader workers [default: 4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    # BN
    parser.add_argument('--bn_momentum', default=0.98, type=float, help='BatchNorm momentum [default: 0.9]')
    parser.add_argument('--bn_decay', default=0.5, type=float, help='BatchNorm momentum decay [default: 0.5]')
    # LR
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_step', type=int,  default=21, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    
    return parser.parse_args()

def test(classifier, criterion, loader, num_class=40):
    num_batches = len(loader)
    classifier = classifier.eval()
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    loss_sum = 0
    total_correct = 0
    total_seen = 0
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        pred = classifier(points)
        loss = criterion(pred, target.long())
        loss_sum += loss
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])
        mean_correct.append(correct.item()/float(points.size()[0])) # overall acc. per batch
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1] # [0:total class acc., 1:num of class seen, 2: average class acc.]
    class_acc = np.mean(class_acc[:,2])
    oa_acc = total_correct/total_seen
    instance_acc = np.mean(mean_correct)
    return oa_acc, instance_acc, class_acc, loss_sum/float(num_batches)

def scale_point_cloud(batch_data, scale_low=0.1, scale_high=10):
    bsize = pc.size(0)
    for i in range(bsize):
        xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
        
        pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
        
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
    return rotated_data


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        load_dir_str = args.log_dir if args.use_ckpt else args.log_dir+'_'+timestr
        experiment_dir = experiment_dir.joinpath(load_dir_str)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/modelnet40/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point*2, split='train',
                                                     normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point*2, split='test',
                                                    normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    '''MODEL LOADING'''
    num_class = 40
    if args.use_ckpt:
        MODEL = importlib.import_module(experiment_dir + args.model)
    else:
        MODEL = importlib.import_module(args.model)
        shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
        shutil.copy(os.path.basename(__file__), str(experiment_dir))


    classifier = MODEL.DensePoint(num_classes = 40, input_channels = 0, use_xyz = True).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        print(model_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    LEARNING_RATE_CLIP = 1e-5
    BNM_CLIP = 0.01
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), BNM_CLIP)
    bnm_scheduler = BNMomentumScheduler(classifier, bnm_lmbd)
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), LEARNING_RATE_CLIP / args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)

    global_epoch = 0
    global_step = 0
    best_oa_acc = 0.0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
            log_string('Learning rate:%f' % (args.learning_rate*lr_lbmd(epoch)))
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)
            log_string('BatchNorm Momentum:%f' % bnm_scheduler.get_momentum())

        num_batches = len(trainDataLoader)
        loss_sum = 0
        classifier = classifier.train(True)
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            points[:,:, 0:3] = rotate_point_cloud(points[:,:, 0:3], angle_range=1) # angle_range * pi
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], scale_low=0.67, scale_high=1.5)
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3], shift_range=0.1)
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, 1024)
            new_fps_idx = fps_idx[:, np.random.choice(1024, args.num_point, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()
            classifier = classifier.train()

            logits = classifier(points)

            pred_choice = logits.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            # num_point = logits.shape[1]
            # logits = logits.view(-1, logits.shape[-1])
            # target = target.view(-1, 1).repeat(1, num_point).view(-1)
            loss = criterion(logits, target.long())
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            
        # lr_scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Epoch Loss: %f, Train Accuracy: %f'% (loss_sum/float(num_batches), train_instance_acc))

        if (epoch+1) % 5 == 0:
            # logger.info('Save model...')
            savepath = str(checkpoints_dir) + ('/model_%s.pth' % (epoch+1))
            log_string('Saving at %s' % (args.log_dir + (': model_%s.pth' % (epoch+1))))
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            
        with torch.no_grad():
            oa_acc, instance_acc, class_acc, eval_loss = test(classifier.eval(), criterion, testDataLoader)

            if (oa_acc >= best_oa_acc):
                best_oa_acc = oa_acc
                best_epoch = epoch + 1

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Eval Loss: %f'% (eval_loss))
            log_string('Test OA: %f, Instance Accuracy: %f, Class Accuracy: %f'% (oa_acc, instance_acc, class_acc))
            log_string('Best OA: %f, Instance Accuracy: %f, Class Accuracy: %f'% (best_oa_acc, best_instance_acc, best_class_acc))

            if (oa_acc >= best_oa_acc):
                time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model' + '.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1


    logger.info('End of training...')


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


if __name__ == '__main__':
    args = parse_args()
    main(args)
