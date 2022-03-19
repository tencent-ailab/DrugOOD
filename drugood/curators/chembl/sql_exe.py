# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
import sqlite3

from mmcv import print_log
from tqdm import tqdm


class SqlExe():
    def __init__(self, cfg):
        '''
        The interface to ChemBL SQL.
        Args:
            cfg: The config object.

        '''
        super(SqlExe, self).__init__()
        self.conn = sqlite3.connect(cfg.source_root)

    def __call__(self, sql, return_cur=True, **kwargs):
        sql = sql.format(**kwargs)
        cur = self.conn.cursor()
        cur.execute(sql)
        if not return_cur:
            res = cur.fetchall()
            return res
        else:
            return cur


class SqlFunctions():
    """Database Enquirer
    """

    def __init__(self, cfg):
        """
        Encapsulate some SQL to complete the SQL query.
        Args:
            cfg: The config object.
        """
        super(SqlFunctions, self).__init__()
        self.sql_exe = SqlExe(cfg)

    def __get_one_col_from_table__(self, table_name, col_name):
        sql = 'select {col_name} from {table_name}'.format(table_name=table_name, col_name=col_name)
        res = self.sql_exe(sql, return_cur=False)
        ans = []
        for item in res:
            ans.append(item[0])
        return ans

    def get_all_assay_ids(self, ):
        return self.__get_one_col_from_table__(table_name='ASSAYS', col_name='ASSAY_ID')

    def get_all_line_all_assay(self, sql_query=None):
        """
        Get all data points from chembl and gather them based on their assay environment
        Args:
            sql_query: query cmd for chembl database
        Returns:
            dict(assay_id：[data]): dict of data gathered from each assay id
        """
        if sql_query is None:
            sql_query = 'select ASSAY_ID, MOLREGNO, STANDARD_VALUE,' \
                        'STANDARD_RELATION, STANDARD_UNITS, STANDARD_TYPE, PCHEMBL_VALUE ' \
                        'from ACTIVITIES'

        queried_results = self.sql_exe(sql_query, False)
        output = {}
        pbar = tqdm(queried_results, total=len(queried_results))
        for idx, line in enumerate(pbar):
            assay_id = line[0]
            if assay_id not in output:
                output[assay_id] = []
            data = {'ASSAY_ID': line[0],
                    'MOLREGNO': line[1],
                    'STANDARD_VALUE': line[2],
                    'STANDARD_RELATION': line[3],
                    'STANDARD_UNITS': line[4],
                    'STANDARD_TYPE': line[5],
                    'PCHEMBL_VALUE': line[6]}
            output[assay_id].append(data)
            if idx % 5000 == 0 or idx == len(queried_results) - 1:
                pbar.set_postfix({'num_vowels': idx})
        return output

    def get_all_lines_in_one_assay(self, assay_id):
        sql = 'select COMPOUND_STRUCTURES.CANONICAL_SMILES, ACTIVITIES.STANDARD_VALUE, ACTIVITIES.STANDARD_TYPE ' \
              'from ACTIVITIES, COMPOUND_STRUCTURES ' \
              'where ACTIVITIES.MOLREGNO = COMPOUND_STRUCTURES.MOLREGNO ' \
              'and ACTIVITIES.ASSAY_ID = "{assay_id}"'.format(assay_id=assay_id)
        res = self.sql_exe(sql, False)
        return res

    def get_smiles_with_molregno(self, molregno):
        sql = 'select CANONICAL_SMILES from COMPOUND_STRUCTURES where MOLREGNO = "{}"'.format(molregno)
        res = self.sql_exe(sql, False)
        if len(res) == 1 and len(res[0]) == 1:
            res = res[0][0]
            return res
        else:
            return None

    def __get_map_in_one_table__(self, table_name, keyname, valuename):
        sql = 'select {keyname}, {valuename} from {table_name}'.format(keyname=keyname,
                                                                       valuename=valuename,
                                                                       table_name=table_name)
        res = self.sql_exe(sql, False)
        res_map = {}
        for line in res:
            key, value = line
            res_map[key] = value
        return res_map

    def get_map_molregno_to_smiles(self, ):
        res = self.__get_map_in_one_table__(table_name='COMPOUND_STRUCTURES',
                                            keyname='MOLREGNO',
                                            valuename='CANONICAL_SMILES')

        return res

    def get_map_assay_id_to_type(self, ):
        return self.__get_map_in_one_table__(table_name='ASSAYS', keyname='ASSAY_ID', valuename='ASSAY_TYPE')

    def statistics_assay_type(self, ):
        assay_type_map = self.get_map_assay_id_to_type()
        type_count = {}
        for _, assay_type in assay_type_map.items():
            if (assay_type not in type_count):
                type_count[assay_type] = 0
            type_count[assay_type] += 1
        print_log('type count:{}'.format(type_count))
        return type_count

    def get_adme_assay_lines(self, ):
        sql = 'select ASSAYS.ASSAY_TYPE,ACTIVITIES.ASSAY_ID, MOLREGNO, STANDARD_VALUE, STANDARD_UNITS, STANDARD_TYPE ' \
              'from ACTIVITIES, ASSAYS where ASSAYS.ASSAY_ID = ACTIVITIES.ASSAY_ID and ASSAYS.ASSAY_TYPE = "A"'
        res = self.sql_exe(sql, True)
        for line in res:
            print(line)

    def statistics_assay_info(self, row_name='STANDARD_UNITS'):
        all_lines = self.get_all_line_all_assay()
        map_count = {}
        for _, lines_cur_assay in all_lines.items():
            for line in lines_cur_assay:
                value = line[row_name]
                if (value not in map_count):
                    map_count[value] = 0
                map_count[value] += 1
        map_count = list(map_count.items())
        map_count = sorted(map_count, key=lambda x: x[1], reverse=True)
        print(map_count)

    def get_protein_squence_of_one_assay(self, assay_id):
        sql = 'select COMPONENT_SEQUENCES.SEQUENCE ' \
              'from ASSAYS, TARGET_COMPONENTS, COMPONENT_SEQUENCES ' \
              'where ASSAYS.TID = TARGET_COMPONENTS.TID ' \
              'and TARGET_COMPONENTS.COMPONENT_ID = COMPONENT_SEQUENCES.COMPONENT_ID ' \
              'and ASSAYS.ASSAY_ID="{assay_id}"'
        res = self.sql_exe(sql, False, assay_id=assay_id)
        if (len(res) == 0):
            return None
        res = res[0][0]
        return res

    def get_assay_target_type(self, assay_id):
        sql = 'select TARGET_DICTIONARY.TARGET_TYPE ' \
              'from TARGET_DICTIONARY,ASSAYS ' \
              'where ASSAYS.TID=TARGET_DICTIONARY.TID ' \
              'and ASSAYS.ASSAY_ID="{assay_id}"'
        res = self.sql_exe(sql=sql, return_cur=False, assay_id=assay_id)
        res = res[0][0]
        return res

    def get_confidence_score_for_assay(self, assay_id):
        sql = 'select CONFIDENCE_SCORE from ASSAYS where ASSAY_ID = "{assay_id}"'
        res = self.sql_exe(sql, False, assay_id=assay_id)
        return res[0][0]

    def get_all_class_id_parent_of_protein(self, ):
        sql = 'select PROTEIN_CLASS_ID, PARENT_ID, CLASS_LEVEL ' \
              'from PROTEIN_CLASSIFICATION'
        res = self.sql_exe(sql, False)
        return res

    def get_all_protein_seq_to_id(self, ):
        sql = 'select COMPONENT_CLASS.PROTEIN_CLASS_ID, COMPONENT_SEQUENCES.SEQUENCE ' \
              'from COMPONENT_CLASS, COMPONENT_SEQUENCES ' \
              'where COMPONENT_CLASS.COMPONENT_ID = COMPONENT_SEQUENCES.COMPONENT_ID'
        res = self.sql_exe(sql, True)
        return res

    def static_table_info(self, table_name, col_name):
        sql = 'select {col_name} from {table_name}'.format(col_name=col_name, table_name=table_name)
        res = self.sql_exe(sql, True)
        count_dict = {}
        count_line = 0
        for line in res:
            count_line += 1
            item = line[0]
            if (item not in count_dict):
                count_dict[item] = 0
            count_dict[item] += 1
        count_list = list(count_dict.items())
        count_list = sorted(count_list, key=lambda x: x[1], reverse=True)
        print('count line:{}'.format(count_line))
        print('count:{}'.format(len(count_list)))
        print(count_list)


if __name__ == '__main__':
    pass
