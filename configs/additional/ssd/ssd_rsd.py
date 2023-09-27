_base_ = '../../ssd/ssd300_coco.py'

data_root = 'data/RSD-COCO-0/' # dataset root

metainfo = {
    'classes': ('airport','helicopter', 'oiltank','plane','warship'),
'palette': [
        (220, 0, 0), (255, 145, 144), (255, 139, 58), (3, 255, 254), (3, 0, 135)
    ]
}

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/images/'),
        ann_file='train/_annotations.coco.json'))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='valid/images/'),
        ann_file='valid/_annotations.coco.json'))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='test/images/'),
        ann_file='test/_annotations.coco.json'))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')


# load COCO pre-trained weight
load_from = './checkpoints/ssd300_coco_20210803_015428-d231a06e.pth'