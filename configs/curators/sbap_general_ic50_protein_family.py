_base_ = ['../_base_/curators/sbap_defaults.py',
          '../_base_/curators/noises/general.py',
          '../_base_/curators/domains/protein_family.py']

path = dict(task=dict(subset="sbap_general_ic50_protein_family"))

noise_filter = dict(assay=dict(measurement_type=['IC50']))
