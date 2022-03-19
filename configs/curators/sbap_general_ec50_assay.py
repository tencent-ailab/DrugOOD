_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/general.py',
          '../_base_/curators/domains/assay.py']

path = dict(task=dict(subset="sbap_general_ec50_assay"))