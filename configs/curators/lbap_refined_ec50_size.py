_base_ = ['../_base_/curators/lbap_defaults.py',
          '../_base_/curators/noises/refined.py',
          '../_base_/curators/domains/size.py']

path = dict(task=dict(subset="lbap_refined_ec50_size"))
