# :fire:DrugOOD:fire::  OOD Dataset Curator and Benchmark for AI Aided Drug Discovery


This is the official implementation of the DrugOOD project, this is the project page: <https://drugood.github.io/>  


## Environment Installation

You can install the conda environment using the drugood.yaml file provided: 

```shell
!git clone https://github.com/tencent-ailab/DrugOOD.git
!cd DrugOOD
!conda env create --name drugood --file=drugood.yaml
!conda activate drugood
```   
Then you can go to the demo at `demo/demo.ipynb` which gives a quick practice on how to use DrugOOD.


## Demo

For a quick practice on using DrugOOD for dataset curation and OOD benchmarking, one can refer to the `demo/demo.ipynb`.   

## Dataset Curator

First, you need to generate the required DrugOOD dataset with our code. The dataset curator currently focusing on  generating datasets from CHEMBL. It supports the following two tasks:

- Ligand Based Affinity Prediction (LBAP).
- Structure Based Affinity Prediction (SBAP).

For OOD domain annotations, it supports the following 5 choices.

- Assay.
- Scaffold.
- Size.
- Protein. (only for SBAP task)
- Protein Family. (only for SBAP task)

For noise annotations, it supports the following three noise levels. Datasets with different
noises are implemented by filters with different levels of strictness.

- Core.
- Refined.
- General.

At the same time, due to the inconvenient conversion between different measurement type (E.g. IC50, EC50, Ki, Potency),   one needs to specify the measurement type when generating the dataset.

### How to Run and Reproduce the 96 Datasets?

Firstly, specifiy the path of CHEMBL database and the directory to save the data in the configuration
file: `configs/_base_/curators/lbap_defaults.py` for LBAP task  or    `configs/_base_/curators/sbap_defaults.py` for SBAP task.   
The `source_root="YOUR_PATH/chembl_29_sqlite/chembl_29.db"` means the path to the 
chembl29 sqllite file.  The `target_root="data/"` specifies the folder to save the generated data.   

Note that you can download the original chembl29 database with sqllite format from `http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29_sqlite.tar.gz`.


The built-in configuration files are located in:    
`configs/curators/`. Here we provide the 96 config files to __reproduce__ the 96 datasets  in our paper.  Meanwhile, 
you can also customize your own datasets by changing the config files.  

Run `tools/curate.py` to generate dataset. Here are some examples:

Generate datasets for the LBAP task, with `assay` as domain, `core` as noise
level, `IC50` as measurement type, `LBAP` as task type.:

```shell
python tools/curate.py --cfg configs/curators/lbap_core_ic50_assay.py
```

Generate datasets for the SBAP task, with `protein` as domain, `refined` as noise level, `EC50` as
measurement type, `SBAP` as task type.:

```shell
python tools/curate.py --cfg configs/curator/sbap_refined_ec50_protein.py
```

## Benchmarking SOTA OOD Algorithms

Currently we support 6 different baseline algorithms:

- ERM
- IRM
- GroupDro
- Coral
- MixUp
- DANN

Meanwhile, we support various GNN backbones:

- GIN
- GCN
- Weave
- ShcNet
- GAT
- MGCN
- NF
- ATi-FPGNN
- GTransformer

And different backbones for protein sequence modeling:

- Bert
- ProteinBert

### How to Run?

Firstly, run the following command to install.

```shell
python setup.py develop
```

Run the LBAP task with ERM algorithm:

```shell
python tools/train.py configs/algorithms/erm/lbap_core_ec50_assay_erm.py
```                                                        

If you would like to run ERM on other datasets, change the corresponding options inside the above
config file. For example,  `ann_file = 'data/lbap_core_ec50_assay.json'`   specifies the input data.  

Similarly, run the SBAP task with ERM algorithm: 

```shell
python tools/train.py configs/algorithms/erm/sbap_core_ec50_assay_erm.py
``` 


## Reference

:smile:If you find this repo is useful, please consider to cite our paper:

```
@ARTICLE{2022arXiv220109637J,
    author = {{Ji}, Yuanfeng and {Zhang}, Lu and {Wu}, Jiaxiang and {Wu}, Bingzhe and {Huang}, Long-Kai and {Xu}, Tingyang and {Rong}, Yu and {Li}, Lanqing and {Ren}, Jie and {Xue}, Ding and {Lai}, Houtim and {Xu}, Shaoyong and {Feng}, Jing and {Liu}, Wei and {Luo}, Ping and {Zhou}, Shuigeng and {Huang}, Junzhou and {Zhao}, Peilin and {Bian}, Yatao},
    title = "{DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery -- A Focus on Affinity Prediction Problems with Noise Annotations}",
    journal = {arXiv e-prints},
    keywords = {Computer Science - Machine Learning, Computer Science - Artificial Intelligence, Quantitative Biology - Quantitative Methods},
    year = 2022,
    month = jan,
    eid = {arXiv:2201.09637},
    pages = {arXiv:2201.09637},
    archivePrefix = {arXiv},
    eprint = {2201.09637},
    primaryClass = {cs.LG}
}
```     

## Disclaimer 
This is not an officially supported Tencent product.