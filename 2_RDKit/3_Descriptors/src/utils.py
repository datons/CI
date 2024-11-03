from rdkit.Chem import Descriptors, MolFromSmiles
import pandas as pd
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Union

class Molecule(BaseModel):
    name: str
    smiles: Optional[str] = None
    properties: Optional[Dict[str, float]] = None

    @validator('smiles')
    def validate_smiles(cls, v):
        if v is not None and not MolFromSmiles(v):
            raise ValueError(f"Invalid SMILES string: {v}")
        return v

class MolecularPropertiesCalculator:
    def __init__(self, input_data: Union[Dict[str, str], List[Union[Dict[str, str], str]]], abbrev=True):
        self.input_data = input_data
        self.abbrev = abbrev
        self.properties = {
            'MolWt': {'full': 'Molecular Weight', 'abbrev': 'MW'},
            'MolLogP': {'full': 'LogP', 'abbrev': 'LogP'},
            'NumHDonors': {'full': 'H-bond Donors', 'abbrev': 'HBD'},
            'NumHAcceptors': {'full': 'H-bond Acceptors', 'abbrev': 'HBA'},
            'NumRotatableBonds': {'full': 'Rotatable Bonds', 'abbrev': 'RB'},
            'NumAromaticRings': {'full': 'Aromatic Rings', 'abbrev': 'AR'},
            'TPSA': {'full': 'TPSA', 'abbrev': 'TPSA'},
            'HeavyAtomCount': {'full': 'Heavy Atom Count', 'abbrev': 'HAC'},
            'FractionCSP3': {'full': 'FractionCSP3', 'abbrev': 'FSP3'},
            'LabuteASA': {'full': 'Labute ASA', 'abbrev': 'ASA'}
        }
        self.molecule_set = []

    def calculate(self, as_df=False, properties=None):
        results = {}
        if isinstance(self.input_data, list):
            for item in self.input_data:
                if isinstance(item, dict):
                    for name, smiles_str in item.items():
                        molecule = MolFromSmiles(smiles_str)
                        if molecule:
                            calculated_properties = self._calculate_for_molecule(molecule, properties)
                            results[name] = {'smiles': smiles_str, 'properties': calculated_properties}
                            self.molecule_set.append(Molecule(name=name, smiles=smiles_str, properties=calculated_properties))
                        else:
                            results[name] = {'smiles': smiles_str, 'properties': None}
                else:
                    # Assume item is a molecule object
                    calculated_properties = self._calculate_for_molecule(item, properties)
                    results[str(item)] = {'smiles': None, 'properties': calculated_properties}
                    self.molecule_set.append(Molecule(name=str(item), properties=calculated_properties))
        elif isinstance(self.input_data, dict):
            for name, smiles_str in self.input_data.items():
                molecule = MolFromSmiles(smiles_str)
                if molecule:
                    calculated_properties = self._calculate_for_molecule(molecule, properties)
                    results[name] = {'smiles': smiles_str, 'properties': calculated_properties}
                    self.molecule_set.append(Molecule(name=name, smiles=smiles_str, properties=calculated_properties))
                else:
                    results[name] = {'smiles': smiles_str, 'properties': None}
        else:
            # Assume input_data is a single molecule object
            calculated_properties = self._calculate_for_molecule(self.input_data, properties)
            results[str(self.input_data)] = {'smiles': None, 'properties': calculated_properties}
            self.molecule_set.append(Molecule(name=str(self.input_data), properties=calculated_properties))

        if as_df:
            return self.to_dataframe()
        else:
            return results

    def _calculate_for_molecule(self, molecule, properties):
        result = {}
        properties_to_calculate = properties if properties else self.properties.keys()
        for desc_name in properties_to_calculate:
            if desc_name not in self.properties:
                raise KeyError(f"Descriptor '{desc_name}' not found in properties.")
            names = self.properties[desc_name]
            try:
                descriptor_func = getattr(Descriptors, desc_name, None)
                if descriptor_func:
                    key = names['abbrev'] if self.abbrev else names['full']
                    result[key] = descriptor_func(molecule)
                else:
                    result[key] = None
            except Exception:
                result[key] = None
        return result

    def to_dataframe(self):
        data = {
            molecule.name: {**molecule.properties, **{'smiles': molecule.smiles}}
            for molecule in self.molecule_set
        }
        return pd.DataFrame.from_dict(data, orient='index')
