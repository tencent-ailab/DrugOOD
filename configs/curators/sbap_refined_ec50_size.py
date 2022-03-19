_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/refined.py',
          '../_base_/curators/domains/size.py']

path = dict(task=dict(subset="sbap_refined_ec50_size"))

noise_filter = dict(assay=dict(assay_target_type=["SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"]))
