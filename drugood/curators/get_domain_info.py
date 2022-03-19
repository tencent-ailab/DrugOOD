# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugood.curators.chembl.protein_family import ProteinFamilyTree


class DomainInfo():
    def __init__(self, cfg, sql_func):
        """
        Args:
            cfg: The config obj.
            sql_func: The SQLFunction obj to get sql information from ChemBL.
        DomainInfo convert the values in the raw data to specific domain description values.
        """
        self.protein_family_getter = ProteinFamilyTree(cfg.get("protein_family_level"), sql_func)

    def scaffold(self, smile):
        """
        Args:
            smile: The smile string in raw data.
        Returns:
            The scaffold string of a smile.
        """
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False)
            return scaffold
        except ValueError:
            print('get scaffold error')
            return 'C'

    def size(self, smile):
        """
        Args:
            smile: The smile string in raw data.
        Returns:
            The number of atoms in a smile.
        """
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            print('GetNumAtoms error, smiles:{}'.format(smile))
            return len(smile)
        number_atom = mol.GetNumAtoms()
        return number_atom

    def assay(self, assay):
        """
        Args:
            assay: The assay ID.
        Returns:
            The assay ID.
        """
        return assay

    def protein(self, protein_seq):
        """

        Args:
            protein_seq: The protein sequence.

        Returns:
            The protein sequence.
        """
        return protein_seq

    def protein_family(self, protein_seq):
        """

        Args:
            protein_seq:The protein sequence.

        Returns:
            The class id of the protein.
        """
        class_id = self.protein_family_getter(protein_seq)
        return class_id


class SortFunc():
    def __init__(self, cfg, sql_func):
        '''

        Args:
            cfg: The config object.
            sql_func: The SQLFunction obj to get sql information from ChemBL.
        Generate the description value of the domain to complete the sorting of the domain.
        '''
        self.domain_info = DomainInfo(cfg, sql_func)

    def domain_value(self, item_domain_data):
        '''

        Args:
            item_domain_data: Single domain data.

        Returns:
            The sorting of domains is done directly according to the value of domain.
        '''
        return item_domain_data[0]

    def domain_capacity(self, item_domain_data):
        '''

        Args:
            item_domain_data: Single domain data.

        Returns:
            Return the number of samples in the domain as the description value of the domain.
        '''
        return len(item_domain_data[1])

    def scaffold_size(self, item_domain_data):
        '''

        Args:
            item_domain_data: Single domain data.

        Returns:
            Returns the size of the scaffold as the description value of the domain.
        '''
        scaffold = item_domain_data[0]
        size_scaffold = self.domain_info.size(scaffold)
        return size_scaffold
