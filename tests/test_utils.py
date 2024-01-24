from copolextractor.utils import name_to_smiles
import pytest
from rdkit import Chem


@pytest.mark.parametrize(
    "name,smiles",
    [
        ("ethene", "C=C"),
        ("ethane", "CC"),
        ("ethanol", "CCO"),
        ("ethanal", "CC=O"),
        ("ethanoic acid", "CC(=O)O"),
        ("ethanamide", "CC(=O)N"),
        ("ethanamine", "CCN"),
        ("ethanenitrile", "CC#N"),
        ("ethanoate", "CC(=O)[O-]"),
        ("ethanoic acid", "CC(=O)O"),
        ("ethanenitrile", "CC#N"),
        ("carbon tetrachloride", "ClC(Cl)(Cl)Cl"),
        ("2-Chloro-1,3-butadiene", "C=CC(=C)Cl"),
        ("Methyl acrylic acid", "CC(=C)C(=O)O"),
        ("Styrol", "C=Cc1ccccc1" ),
        ("Styren", "C=Cc1ccccc1")
    ],
)
def test_name_to_smiles(name, smiles):
    mol_expected = Chem.MolFromSmiles(smiles)
    mol_result = Chem.MolFromSmiles(name_to_smiles(name))
    assert Chem.MolToSmiles(mol_expected) == Chem.MolToSmiles(mol_result)
