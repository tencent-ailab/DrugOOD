# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from drugood.utils.smile_to_dgl import smile2graph


class Filter(object):
    """Compose a filter pipeline with a sequence of sub filters.

    Args:
        cfg (Dict[]):
            Either config dicts of filters.
    """

    def __init__(self, cfg):
        """
        The Base class of filter.
        Args:
            cfg: The config object.
        """
        self.filter_func = []
        for (filter, config) in cfg.items():
            one_func = self.__getattribute__(filter)
            self.filter_func.append(one_func)

    def __call__(self, one_data):
        for func in self.filter_func:
            res = func(one_data)
            if not res:
                return False
        return True


class AssayFilter(Filter):
    def __init__(self, cfg, sql_func):
        """
        The Filter for assay.
        Args:
            cfg: The config object.
            sql_func: The object of SQLFunction.
        """
        self.cfg = cfg
        self.sql_func = sql_func
        super(AssayFilter, self).__init__(cfg=cfg)

    def measurement_type(self, data):
        for case in data:
            index_type = case['STANDARD_TYPE']
            if index_type not in self.cfg["measurement_type"]:
                return False
        return True

    def assay_value_units(self, data):
        for case in data:
            units = case['STANDARD_UNITS']
            if units not in self.cfg["assay_value_units"]:
                return False
        return True

    def molecules_number(self, data):
        number_m = len(data)
        if self.cfg["molecules_number"][0] <= number_m <= self.cfg["molecules_number"][1]:
            return True
        else:
            return False

    def assay_target_type(self, data):
        assay_id = data[0]['ASSAY_ID']
        target_type = self.sql_func.get_assay_target_type(assay_id)
        if target_type in self.cfg["assay_target_type"]:
            return True
        else:
            return False

    def confidence_score(self, data):
        assay_id = data[0]['ASSAY_ID']
        confidence_score = self.sql_func.get_confidence_score_for_assay(assay_id)
        confidence_score = int(confidence_score)
        if self.cfg["confidence_score"] is None:
            return True
        elif confidence_score >= self.cfg["confidence_score"]:
            return True
        else:
            return False
        


class SampleFilter(Filter):
    def __init__(self, cfg, mol_id_to_smile):
        """
        The Filter for samples.
        Args:
            cfg: The config object.
            mol_id_to_smile:
        """
        self.cfg = cfg
        self.mol_id_to_smile = mol_id_to_smile
        super(SampleFilter, self).__init__(cfg=cfg)

    def filter_none(self, case):
        for item in case.values():
            if item is None:
                return False
        return True

    def smile_exist(self, case):
        molregno = case['MOLREGNO']
        if molregno in self.mol_id_to_smile:
            return True
        else:
            return False

    def smile_legal(self, case):
        molregno = case['MOLREGNO']
        smile = self.mol_id_to_smile[molregno]
        graph = smile2graph(smile)
        if graph is None or graph.num_edges() == 0 or graph.num_nodes() == 0:
            return False
        else:
            return True

    def value_relation(self, case):
        relation = case['STANDARD_RELATION']
        if relation in self.cfg["value_relation"]:
            return True
        else:
            return False
