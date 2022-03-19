_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/general.py',
          '../_base_/curators/domains/scaffold.py']

path = dict(task=dict(subset="sbap_general_ki_scaffold"))

noise_filter = dict(assay=dict(measurement_type=['Ki']))
