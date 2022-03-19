# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')
runner = dict(type='EpochBasedRunner', max_epochs=50)
# evaluation config
evaluation = dict(metric=['accuracy', 'auc'])
