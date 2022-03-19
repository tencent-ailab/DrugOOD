# data path
path = dict(
    task=dict(type="sbap", subset="sbap_core_ec50_assay"),
    source_root="/apdcephfs/share_1364275/xluzhang/chembl_29_sqlite/chembl_29.db",
    target_root="data/"
)

# uncertainty
uncertainty = dict(delta={'<': -1, '<=': -1, '>': 1, '>=': 1})

# adaptive cls label
classification_threshold = dict(
    lower_bound=4,
    upper_bound=6,
    fix_value=5
)

# train/val/test
fractions = dict(
    train_fraction_ood=0.6,
    val_fraction_ood=0.2,
    iid_train_sample_fractions=0.6,
    iid_val_sample_fractions=0.2
)
