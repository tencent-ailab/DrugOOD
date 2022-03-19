_base_ = ['../_base_/curators/lbap_defaults.py',
          '../_base_/curators/noises/general.py',
          '../_base_/curators/domains/size.py']

path = dict(task=dict(subset="lbap_general_potency_size"))

noise_filter = dict(assay=dict(measurement_type=['Potency']))