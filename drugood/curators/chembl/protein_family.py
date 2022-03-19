class ProteinFamilyTree():
    """query protein's level identifier at a specific family level
    Args:
    protein_family_level (int):
        Specific protein family level.
    sql_func:
        Queried Database
    """

    def __init__(self, protein_family_level, sql_func):
        '''
        Output the class of a protein in a protein family.
        Args:
            protein_family_level: Hyperparameter that controls which layer in the multi-level protein classification to output.
            sql_func: The object of SQLFunction.
        '''
        super(ProteinFamilyTree, self).__init__()
        self.id_target_level = protein_family_level
        link_nodes = sql_func.get_all_class_id_parent_of_protein()
        dict_id_to_parent_level = {}
        for item in link_nodes:
            cur_id, parent_id, level = item
            dict_id_to_parent_level[cur_id] = (parent_id, level)
        self.dict_id_to_parent_level = dict_id_to_parent_level

        dict_protein_seq_to_classid = {}
        for item in sql_func.get_all_protein_seq_to_id():
            class_id, protein_seq = item
            dict_protein_seq_to_classid[protein_seq] = class_id
        self.dict_protein_seq_to_classid = dict_protein_seq_to_classid

    def get_target_level_class_id(self, class_id_cur_level):
        cur_level = self.dict_id_to_parent_level[class_id_cur_level][1]
        while True:
            if cur_level == self.id_target_level:
                break
            class_id_cur_level = self.dict_id_to_parent_level[class_id_cur_level][0]
            cur_level -= 1
        dict_level = self.dict_id_to_parent_level[class_id_cur_level][1]
        assert dict_level == self.id_target_level, \
            'dict_level:{}, target level:{}'.format(dict_level, self.id_target_level)
        return class_id_cur_level

    def __call__(self, protein_seq):
        class_id = self.dict_protein_seq_to_classid[protein_seq]
        target_level_class_id = self.get_target_level_class_id(class_id)
        return target_level_class_id
