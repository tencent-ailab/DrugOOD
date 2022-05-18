import json
import os.path as osp
import random

import numpy as np
import tqdm
from mmcv import Config, print_log

from drugood.core import make_dirs
from drugood.curators.chembl.filter import AssayFilter, SampleFilter
from drugood.curators.chembl.sql_exe import SqlFunctions
from drugood.curators.get_domain_info import DomainInfo, SortFunc


class GenericCurator(object):
    """Base curator.
    Args:
        cfg (dict): config
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.statistics = {}
        self.sql = SqlFunctions(self.cfg.get("path", None))
        self.mapping = self.sql.get_map_molregno_to_smiles()
        self.set_filters()

    def set_filters(self):
        """ initial assay and sample filter

        """
        filter_cfg = self.cfg.get("noise_filter", None)
        self.assay_filter = AssayFilter(filter_cfg.get("assay"), self.sql)
        self.sample_filter = SampleFilter(filter_cfg.get("sample"), self.mapping)

    def data_loading(self, sql_query=None):
        """Get all data points from chembl and gather them based on their assay environment
            Args:
                sql_query: query cmd for chembl database
            Returns:
                dict(assay_id：[data]): dict of data gathered from each assay id
        """
        if sql_query is None:
            sql_query = 'select ASSAY_ID, MOLREGNO, STANDARD_VALUE,' \
                        'STANDARD_RELATION, STANDARD_UNITS, STANDARD_TYPE, PCHEMBL_VALUE ' \
                        'from ACTIVITIES'
        queried_results = self.sql.sql_exe(sql_query, False)
        output = {}
        pbar = tqdm.tqdm(queried_results, desc="Data Loading", total=len(queried_results))
        for idx, case in enumerate(pbar):
            assay = case[0]
            if assay not in output:
                output[assay] = []
            data = {
                'ASSAY_ID': case[0],
                'MOLREGNO': case[1],
                'STANDARD_VALUE': case[2],
                'STANDARD_RELATION': case[3],
                'STANDARD_UNITS': case[4],
                'STANDARD_TYPE': case[5],
                'PCHEMBL_VALUE': case[6]
            }
            output[assay].append(data)
        return output

    def noise_filtering(self, data):
        """ Exclude unwanted data samples based on the filters
            Args:
                data (Dict(List)):
            Returns:
                output (List(Dict))
        """
        task_cfg = self.cfg.get("path").get("task")
        output = []
        pbar = tqdm.tqdm(data.items(), desc="Noise Filtering", total=len(data.items()))
        for idx, case in enumerate(pbar):
            assay, samples = case
            if self.assay_filter(samples):
                if task_cfg.type == 'sbap':
                    protein_seq = self.sql.get_protein_squence_of_one_assay(assay_id=assay)
                    if protein_seq is None:
                        pass
                for sample in samples:
                    if self.sample_filter(sample):
                        molregno = sample['MOLREGNO']
                        smile = self.mapping[molregno]
                        res = {'smiles': smile,
                               'reg_label': sample['PCHEMBL_VALUE'],
                               'assay_id': assay,
                               'relation': sample['STANDARD_RELATION']}
                        if task_cfg.type == 'sbap':
                            res['protein'] = protein_seq
                        output.append(res)
        return output

    def process_uncertain_value_offset(self, data):
        """ Process uncertain offset
            Args:
                data (List(Dict)):
            Returns:
                output (List(Dict))
        """
        uncertainty_delta = self.cfg.get("uncertainty").get("delta")
        for case in tqdm.tqdm(data, total=len(data), desc="Uncertainty Processing"):
            relation = case['relation']
            if relation in uncertainty_delta:
                delta = uncertainty_delta[relation]
                case['reg_label'] = case['reg_label'] + delta
            del case['relation']
        return data

    def process_multiple_measurement(self, data):
        """ Process cases with multiple measurement values
            Args:
                data (List(Dict)):
            Returns:
                output (List(Dict))
        """
        sbap = True if self.cfg.path.task.type == 'sbap' else False
        key_to_data = {}
        for case in data:
            if not sbap:
                key = case['smiles']
            else:
                key = case['smiles'] + case['protein']
            if key not in key_to_data:
                key_to_data[key] = []
            key_to_data[key].append(case)
        output = []
        for key, lines in key_to_data.items():
            if len(lines) == 1:
                output += lines
            else:
                all_values_cur_smiles = []
                for one_line in lines:
                    all_values_cur_smiles.append(one_line['reg_label'])
                mean_value = np.mean(all_values_cur_smiles)
                new_line = {'smiles': lines[0]['smiles'], 'reg_label': mean_value, 'assay_id': lines[0]['assay_id']}
                if sbap:
                    new_line['protein'] = lines[0]['protein']
                output.append(new_line)
        return output

    def uncertainty_processing(self, data):
        """ Process cases with uncertainty
            Args:
                data (List(Dict)):
            Returns:
                output (List(Dict))
        """
        # deal with uncertainty value offset problem
        data = self.process_uncertain_value_offset(data)
        # deal with multiple measurement
        data = self.process_multiple_measurement(data)
        return data

    def classification_label_generating(self, data):
        """ Generate classification label for data
            Args:
                data (List(Dict)):
            Returns:
                output (List(Dict))
        """

        all_values = [case['reg_label'] for case in data]
        all_values = sorted(all_values)
        median = all_values[int(len(all_values) * 0.5)]
        if self.cfg.classification_threshold.lower_bound <= median <= self.cfg.classification_threshold.upper_bound:
            thr = median
        else:
            thr = self.cfg.classification_threshold.fix_value
        self.statistics['thr_for_cls'] = thr
        positive_samples = 0
        negative_samples = 0
        for line in data:
            if line['reg_label'] >= thr:
                positive_samples += 1
                cls_label = 1
            else:
                negative_samples += 1
                cls_label = 0
            line['cls_label'] = cls_label
        self.statistics['positive_samples'] = positive_samples
        self.statistics['negative_samples'] = negative_samples
        self.statistics['positive_rate'] = positive_samples / (positive_samples + negative_samples)
        return data

    def data_splitting(self, data):
        """ Make splits of train, val (id, ood), test (id, ood) based on domain split.
            Args:
                data (List(Dict)):
            Returns:
                output (List(Dict))
        """

        assert isinstance(data, list)
        assert ('smiles' in data[0] or 'protein' in data[0]) and 'cls_label' in data[0]
        domain_cfg = self.cfg.domain

        domain_info_funcs_set = DomainInfo(self.cfg, self.sql)
        domain_func = getattr(domain_info_funcs_set, domain_cfg.domain_name)

        data_each_domain = {}
        for case in data:
            value_for_generating_domain = case[domain_cfg.domain_generate_field]
            domain_value = domain_func(value_for_generating_domain)
            if domain_value not in data_each_domain:
                data_each_domain[domain_value] = []
            data_each_domain[domain_value].append(case)

        for domain_id, (domain_value, lines_cur_domain) in enumerate(data_each_domain.items()):
            for one_line in lines_cur_domain:
                one_line['domain_id'] = domain_id

        list_domain_data = list(data_each_domain.items())

        assert domain_cfg.sort_order in ['descend', 'ascend']
        reverse = True if domain_cfg.sort_order == 'descend' else False
        sort_func_sets = SortFunc(domain_cfg, self.sql)
        list_domain_data = sorted(list_domain_data,
                                  key=getattr(sort_func_sets, domain_cfg.sort_func),
                                  reverse=reverse)
        fraction_cfg = self.cfg.fractions
        train_fraction_ood = fraction_cfg.train_fraction_ood
        val_fraction_ood = fraction_cfg.val_fraction_ood

        total_number = sum([len(item[1]) for item in list_domain_data])
        target_train_number = int(total_number * train_fraction_ood)
        target_val_number = int(total_number * val_fraction_ood)
        number_train_domains = 0
        number_val_domains = 0
        count = 0
        for domain_id, lines_one_domain in list_domain_data:
            if count < target_train_number:
                count += len(lines_one_domain)
                number_train_domains += 1
            elif count < target_train_number + target_val_number:
                count += len(lines_one_domain)
                number_val_domains += 1

        domain_train = list_domain_data[:number_train_domains]
        domain_val = list_domain_data[number_train_domains:number_train_domains + number_val_domains]
        domain_test = list_domain_data[number_train_domains + number_val_domains:]

        self.statistics['train domain number'] = len(domain_train)
        self.statistics['val domain number'] = len(domain_val)
        self.statistics['test domain number'] = len(domain_test)
        # split domain_train for iid evaluation
        iid_train_sample_fractions = fraction_cfg.iid_train_sample_fractions
        iid_val_sample_fractions = fraction_cfg.iid_val_sample_fractions
        train = []
        iid_val = []
        iid_test = []
        for (domain_id, data_cur_domain) in domain_train:
            random.shuffle(data_cur_domain)
            len_train = int(len(data_cur_domain) * iid_train_sample_fractions)
            len_val = int(len(data_cur_domain) * iid_val_sample_fractions)
            train_cur_domain = data_cur_domain[:len_train]
            iid_val_cur_domain = data_cur_domain[len_train:len_train + len_val]
            iid_test_cur_domain = data_cur_domain[len_train + len_val:]
            train += train_cur_domain
            iid_val += iid_val_cur_domain
            iid_test += iid_test_cur_domain

        ood_val = []
        ood_test = []
        for item in domain_val:
            ood_val += item[1]
        for item in domain_test:
            ood_test += item[1]
        split = {'train': train, 'ood_val': ood_val, 'ood_test': ood_test, 'iid_val': iid_val, 'iid_test': iid_test}

        self.statistics['train datapoints'] = len(train)
        self.statistics['ood_val datapoints'] = len(ood_val)
        self.statistics['ood_test datapoints'] = len(ood_test)
        self.statistics['iid_val datapoints'] = len(iid_val)
        self.statistics['iid_test datapoints'] = len(iid_test)

        self.statistics['train domain number'] = len(set([item["domain_id"] for item in train]))
        self.statistics['ood_val domain number'] = len(set([item["domain_id"] for item in ood_val]))
        self.statistics['ood_test domain number'] = len(set([item["domain_id"] for item in ood_test]))
        self.statistics['iid_val domain number'] = len(set([item["domain_id"] for item in iid_val]))
        self.statistics['iid_test domain number'] = len(set([item["domain_id"] for item in iid_test]))

        return split

    def data_saving(self, data):
        """ Save the curated data.
            Args:
                data (List(Dict)):
        """
        assert isinstance(data, dict)
        path_cfg = self.cfg.get("path")
        save_dir = osp.join(path_cfg.target_root, '{}.json'.format(path_cfg.task.subset))
        make_dirs(osp.dirname(save_dir))

        save_obj = dict(
            cfg=dict(self.cfg),
            split=data,
            statistics=self.statistics
        )

        with open(save_dir, 'w') as f:
            json.dump(save_obj, f)
        print_log(f"data saved to {save_dir}")

    def statistics_reporting(self):
        print_log('-' * 60)
        from prettytable import PrettyTable
        table = PrettyTable(['split', 'domain number', 'sample number '], title="Statistics of curated ood dataset")
        table.add_row(['train', self.statistics['train domain number'], self.statistics['train datapoints']])
        table.add_row(['iid_val', self.statistics['iid_val domain number'], self.statistics['iid_val datapoints']])
        table.add_row(['ood_val', self.statistics['ood_val domain number'], self.statistics['ood_val datapoints']])
        table.add_row(['iid_test', self.statistics['iid_test domain number'], self.statistics['iid_test datapoints']])
        table.add_row(['ood_test', self.statistics['ood_test domain number'], self.statistics['ood_test datapoints']])
        print_log(table)
