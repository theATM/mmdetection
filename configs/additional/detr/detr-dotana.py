_base_ =  '../../detr/detr_r50_8xb2-150e_coco.py'


data_root = 'data/RSD-COCO-DOTANA-T0/' # dataset root

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 25
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)


metainfo = {
    'classes': ('airport','helicopter', 'oiltank','plane','warship'),
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/instances_train.json'))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/val/'),
        ann_file='annotations/instances_val.json'))

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/test/'),
        ann_file='annotations/instances_test.json'))

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')

test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

model = dict(bbox_head=dict(num_classes=5))

# load COCO pre-trained weight
load_from = './checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

# only keep latest 2 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))