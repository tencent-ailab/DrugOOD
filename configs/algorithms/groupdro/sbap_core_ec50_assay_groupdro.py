_base_ = ['../../_base_/schedules/classification.py', '../../_base_/default_runtime.py']

# transform
train_pipeline = [
    dict(
        type="SmileToGraph",
        keys=["input"]
    ),
    dict(
        type='Collect',
        keys=['input', 'aux_input', 'gt_label', 'group']
    )
]
test_pipeline = [
    dict(
        type="SmileToGraph",
        keys=["input"]
    ),
    dict(
        type='Collect',
        keys=['input', 'aux_input', 'gt_label', 'group']
    )]

# dataset
dataset_type = "SBAPDataset"
ann_file = 'data/sbap_core_ec50_assay.json'

tokenizer = dict(
    type="SeqToToken",
    model="bert-base-uncased"
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        split="train",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        tokenizer=tokenizer,
        sample_mode="group",
        sample_config=dict(
            uniform_over_groups=None,
            n_groups_per_batch=4,
            distinct_groups=True
        )

    ),
    ood_val=dict(
        split="ood_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        tokenizer=tokenizer,
        rule="greater",
        save_best="accuracy"
    ),
    iid_val=dict(
        split="iid_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        tokenizer=tokenizer
    ),
    ood_test=dict(
        split="ood_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        tokenizer=tokenizer,
    ),
    iid_test=dict(
        split="iid_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        tokenizer=tokenizer
    ),
)
model = dict(
    type="GroupDRO",
    tasker=dict(
        type='Classifier',
        backbone=dict(
            type='GIN',
            num_node_emb_list=[39],
            num_edge_emb_list=[10],
            num_layers=4,
            emb_dim=128,
            readout='sum',
            JK='last',
            dropout=0.1,
        ),
        aux_backbone=dict(
            type='Bert',
            model="bert-base-uncased",
        ),
        neck=dict(type="Concatenate"),
        head=dict(
            type='LinearClsHead',
            num_classes=2,
            in_channels=128 + 768,
            loss=dict(
                type='CrossEntropyLoss',
                reduction='none',
            )
        )
    ),
    group_dro_step_size=0.001,
    num_groups=55,
)
