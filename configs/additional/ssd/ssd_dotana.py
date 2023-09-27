_base_ = '../../ssd/ssd300_coco.py'

data_root = 'data/RSD-COCO-DOTANA-T0/' # dataset root

metainfo = {
    'classes': ('airport','helicopter', 'oiltank','plane','warship'),
    'palette': [   (220, 0, 0), (255, 145, 144), (255, 139, 58), (3, 255, 254), (3, 0, 135) ]
}

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/train/'),
        ann_file='annotations/instances_train.json'))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/val/'),
        ann_file='annotations/instances_val.json'))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='images/test/'),
        ann_file='annotations/instances_test.json'))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')


# load COCO pre-trained weight
load_from = './checkpoints/ssd300_coco_20210803_015428-d231a06e.pth'