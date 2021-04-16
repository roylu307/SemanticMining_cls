import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
from pointnet2_ops import pointnet2_utils
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='smnet_part_seg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=500, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--use_ckpt', action='store_true', default=False, help='model name [default: pointnet_cls]')
    parser.add_argument('--log_dir', type=str, default='smnet_part_seg', help='Log path [default: None]')
    # DATA
    parser.add_argument('--num_point', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--num_worker', type=int, default=16, help='number of dataloader workers [default: 4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    # BN
    parser.add_argument('--bn_momentum', default=0.98, type=float, help='BatchNorm momentum [default: 0.9]')
    parser.add_argument('--bn_decay', default=0.7, type=float, help='BatchNorm momentum decay [default: 0.5]')
    # LEARNING RATE
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--decay_step', type=int,  default=12, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.8, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()

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
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        load_dir_str = args.log_dir if args.use_ckpt else args.log_dir+'_'+timestr
        experiment_dir = experiment_dir.joinpath(args.log_dir+'_'+timestr)

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

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=2048, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=args.num_worker)
    TEST_DATASET = PartNormalDataset(root = root, npoints=2048, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=args.num_worker)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16
    num_part = 50
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy(os.path.basename(__file__), str(experiment_dir))

    input_channels = 3 if args.normal else 0
    classifier = MODEL.DensePoint(num_classes = 50, input_channels = input_channels, use_xyz = True).cuda()
    # classifier = torch.nn.DataParallel(classifier)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
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
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    BNM_CLIP = 0.01
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), BNM_CLIP)
    bnm_scheduler = BNMomentumScheduler(classifier, bnm_lmbd)
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), LEARNING_RATE_CLIP / args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)


    best_acc = 0
    global_epoch = 0
    global_step = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
            log_string('Learning rate:%f' % (args.learning_rate*lr_lbmd(epoch)))
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)
            log_string('BatchNorm Momentum:%f' % bnm_scheduler.get_momentum())

        mean_correct = []

        num_batches = len(trainDataLoader)
        loss_sum = 0
        classifier = classifier.train()
        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], scale_low=0.67, scale_high=1.5)
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3], shift_range=0.1)
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()

            points, target = points.cuda(), target.cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, 2048)
            # fps_idx = fps_idx[:, np.random.choice(2048, args.num_point, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            target = pointnet2_utils.gather_operation(target.unsqueeze(-1).float().transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous().squeeze(-1).int().long()

            optimizer.zero_grad()
            seg_pred = classifier(points, to_categorical(label, 16))

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, target)
            loss_sum += loss

            loss.backward()
            optimizer.step()
            global_step += 1

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
            test_metrics = {}
            loss_sum = 0
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                # points = points.transpose(2, 1)
                classifier = classifier.eval()
                seg_pred = classifier(points, to_categorical(label, num_classes))

                loss = criterion(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0].long())
                loss_sum += loss

                cur_pred_val_logits = seg_pred.cpu().data.numpy()

                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            log_string('Eval Loss: %f'% (loss_sum/float(num_batches)))
            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        global_epoch+=1

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

