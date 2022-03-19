_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/core.py',
          '../_base_/curators/domains/assay.py']

noise_filter = dict(assay=dict(assay_target_type=["SINGLE PROTEIN"]))
