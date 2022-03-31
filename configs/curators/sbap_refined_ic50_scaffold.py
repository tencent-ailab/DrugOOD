_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/refined.py',
          '../_base_/curators/domains/scaffold.py']

path = dict(task=dict(subset="sbap_refined_ic50_scaffold"))

noise_filter = dict(
        assay=dict(
            measurement_type=['IC50'],
            assay_target_type=["SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY"]))
