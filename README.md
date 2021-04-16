# Semantic Feature Mining fpr 3D Object Classification and Segmentation

This repo provides the Pytorch implementation of Semantic Feature Mining fpr 3D Object Classification and Segmentation accepted by ICRA2021.

The code is heavily built on [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/b4e79513391c11e98df30d3241a0a24ed3cb3a2a) by Benny 

### Performance
| Model | Accuracy |
|--|--|
| smnet_9layer |  93.2|


## Installation
Install PointNet++ library
```
$ pip install -r requirements.txt
```
The codes are tested on:

- OS: Ubuntu 16.04/18.04
- Python: 3.7.6
- PyTorch: 1.5.1
- CUDA: 10.0
- CUDNN: 7.6.5

## Train and Evaluation
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

Train with model in `./models`. eg. smnet_9layer

```
python train_cls.py --model smnet_9layer --log_dir smnet_9layer --batch_size 72
python test_cls.py --log_dir smnet_9layer
```

Evaluation
```

python test_cls.py --log_dir smnet_9layer
```


## Reference
[yanx27/pointnet_pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/b4e79513391c11e98df30d3241a0a24ed3cb3a2a)<br>
[erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)

