_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

data_root = 'data/RSD-COCO-DOTANA-T0/' # dataset root

metainfo = {
    'classes': ('airport','helicopter', 'oiltank','plane','warship'),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/instances_train.json'))

val_dataloader = dict(
    batch_size=4,
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

model = dict(roi_head=dict(bbox_head=dict(num_classes=5)))

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=25, val_interval=1)


# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)