_base_ =  '../../detr/detr_r50_8xb2-150e_coco.py'


data_root = 'data/RSD-COCO-0/' # dataset root

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 50
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
        data_prefix=dict(img='train/images/'),
        ann_file='train/_annotations.coco.json'))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='valid/images/'),
        ann_file='valid/_annotations.coco.json'))

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='test/images/'),
        ann_file='test/_annotations.coco.json'))

val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')

test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')

model = dict(bbox_head=dict(num_classes=5))

# load COCO pre-trained weight
load_from = './checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'