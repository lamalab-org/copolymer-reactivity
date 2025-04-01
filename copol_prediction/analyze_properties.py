from copolextractor.mongodb_storage import CoPolymerDB


def analyze_smiles_counts():
    db = CoPolymerDB()

    all_monomer_names = set()  # All unique monomer names
    converted_names = set()  # Names with SMILES
    unconverted_names = set()  # Names without SMILES
    existing_smiles = set()

    # Track mappings for verification
    name_to_smiles = {}  # {name: smiles}

    for entry in db.collection.find():
        monomers = entry.get('monomers', [])

        # Process up to 2 monomers
        for i, monomer in enumerate(monomers[:2]):
            if isinstance(monomer, dict):
                name = monomer.get('monomer1') or monomer.get('monomer2')
            else:
                name = monomer if monomer not in ['monomer1', 'monomer2'] else None

            if name:
                all_monomer_names.add(name)
                smiles = entry.get(f"monomer{i + 1}_s")

                if smiles:
                    converted_names.add(name)
                    existing_smiles.add(smiles)
                    name_to_smiles[name] = smiles
                else:
                    unconverted_names.add(name)

    # Names can appear in both sets if they're sometimes converted and sometimes not
    truly_unconverted = unconverted_names - converted_names

    print("\nDetailed SMILES Analysis:")
    print(f"Total unique monomer names: {len(all_monomer_names)}")
    print(f"Names with SMILES: {len(converted_names)}")
    print(f"Names without SMILES: {len(truly_unconverted)}")
    print(f"Unique existing SMILES: {len(existing_smiles)}")
    print(f"Missing SMILES conversions: {len(all_monomer_names) - len(converted_names)}")

    print("\nVerification:")
    print(f"Each name should have exactly one SMILES:")
    for name in all_monomer_names:
        smiles_count = len(
            set(s for s in [entry.get('monomer1_s') for entry in db.collection.find({'monomers.0': name})] +
                [entry.get('monomer2_s') for entry in db.collection.find({'monomers.1': name})]
                if s is not None))
        if smiles_count > 1:
            print(f"- {name}: {smiles_count} different SMILES values")

    return {
        "total_unique_names": len(all_monomer_names),
        "names_with_smiles": len(converted_names),
        "names_without_smiles": len(truly_unconverted),
        "unique_smiles": len(existing_smiles),
        "missing_conversions": len(all_monomer_names) - len(converted_names),
        "unconverted_names": sorted(list(truly_unconverted))
    }


if __name__ == "__main__":
    results = analyze_smiles_counts()
    print("\nNames without SMILES conversion:")
    for name in results["unconverted_names"]:
        print(f"- {name}")