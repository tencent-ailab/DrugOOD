_base_ = ['../_base_/curators/lbap_defaults.py',
          '../_base_/curators/noises/core.py',
          '../_base_/curators/domains/assay.py']

path = dict(task=dict(subset="lbap_core_ic50_assay"))

noise_filter = dict(assay=dict(measurement_type=['IC50']))
