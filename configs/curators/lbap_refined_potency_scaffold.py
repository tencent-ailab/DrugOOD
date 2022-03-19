_base_ = ['../_base_/curators/lbap_defaults.py',
          '../_base_/curators/noises/refined.py',
          '../_base_/curators/domains/scaffold.py']

path = dict(task=dict(subset="lbap_refined_potency_scaffold"))

noise_filter = dict(assay=dict(measurement_type=['Potency']))

