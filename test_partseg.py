import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
from pointnet2_ops import pointnet2_utils
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


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
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='smnet_partseg', help='Experiment root')
    parser.add_argument('--model', type=str, default='best_model', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=10, help='Aggregate segmentation scores with voting [default: 3]')
    parser.add_argument('--num_repeat', type=int, default=100, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--from_epoch', type=int, default=100, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--eval_all', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--save_output', '-s', help='Save results as txt', default=False)
    return parser.parse_args()

class PointcloudScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size(0)
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

def test(classifier, testDataLoader,output_txt_dir, num_part=50, num_classes=16, vote_num=1):
    pc_rand_scale = PointcloudScale()
    classifier = classifier.eval()
    with torch.no_grad():
        test_metrics = {}
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

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

            new_fps_idx = pointnet2_utils.furthest_point_sample(points, 2400)
            new_fps_idx = new_fps_idx[:, np.random.choice(2400, args.num_point, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()
            target = pointnet2_utils.gather_operation(target.unsqueeze(-1).float().transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous().squeeze(-1).int().long()

            cur_batch_size, NUM_POINT, _ = points.size()

            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            for v in range(vote_num):
                new_points = points
                if v > 0:
                    new_points.data = pc_rand_scale(new_points.data)

                seg_pred = classifier(new_points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / vote_num
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

            # print(cur_pred_val.shape)
            # print(points.shape)
            # print(target.shape)
            save_result_txt(points.cpu(), target, cur_pred_val, label.cpu(), output_txt_dir)

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


        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        return test_metrics, shape_ious


def save_result_txt(points, labels, pred, category, output_txt_dir):
    labels = np.expand_dims(np.array(labels), axis=-1)
    pred = np.expand_dims(np.array(pred), axis=-1)
    points = np.array(points)
    category = np.array(category)
    # print(category.shape)

    output = np.concatenate((points, labels, pred), axis=-1)
    counter = np.zeros(16, dtype=int)
    for i in range(points.shape[0]):
        
        filename = os.path.join(BASE_DIR,output_txt_dir, str(category[i][0]),  str(counter[category[i]]) + '.txt')
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        file = open(filename, 'w')
        np.savetxt(file,output[i,:,:])
        file.close()
        counter[category[i]] += 1







def main(args):
    
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir
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

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root = root, npoints=4096, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    print(experiment_dir)


    output_txt_dir = (os.path.join(experiment_dir, 'results'))





    input_channels = 3 if args.normal else 0
    classifier = MODEL.DensePoint(num_classes = 50, input_channels = input_channels, use_xyz = True).cuda()
    # classifier = torch.nn.DataParallel(classifier)

    checkpoint = torch.load(model_dir + args.model + '.pth')

    # original saved file with DataParallel
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '') # removing ‘.moldule’ from key
        new_state_dict[name] = v
    # load params
    classifier.load_state_dict(new_state_dict)


    # classifier.load_state_dict(checkpoint['model_state_dict'])

    pc_rand_scale = PointcloudScale()

    best_acc = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for i in range(args.num_repeat):
        test_metrics, shape_ious = test(classifier, testDataLoader, num_part=num_part, num_classes=num_classes, vote_num=args.num_votes, output_txt_dir=output_txt_dir)

        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))

        log_string('Repeat (%3d/%3d)' % (i+1, args.num_repeat))
        log_string('Accuracy is: %.5f'%test_metrics['accuracy'])
        log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
        log_string('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)

if __name__ == '__main__':
    args = parse_args()
    main(args)

