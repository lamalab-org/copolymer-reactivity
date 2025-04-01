from copolextractor.mongodb_storage import CoPolymerDB
from collections import Counter


def show_top_monomers():
    """Show overview of top 5 most frequent monomers with names and SMILES"""
    db = CoPolymerDB()

    # Count monomers and collect their names
    monomer_count = Counter()
    monomer_details = {}

    # Find all entries with monomer information
    for entry in db.collection.find({}, {
        'monomer1_s': 1,
        'monomer2_s': 1,
        'monomers': 1
    }):
        # Process monomer 1
        if 'monomer1_s' in entry and entry['monomer1_s']:
            smiles = entry['monomer1_s']
            monomer_count[smiles] += 1
            if smiles not in monomer_details and 'monomers' in entry and len(entry['monomers']) > 0:
                monomer_details[smiles] = entry['monomers'][0]

        # Process monomer 2
        if 'monomer2_s' in entry and entry['monomer2_s']:
            smiles = entry['monomer2_s']
            monomer_count[smiles] += 1
            if smiles not in monomer_details and 'monomers' in entry and len(entry['monomers']) > 1:
                monomer_details[smiles] = entry['monomers'][1]

    # Get top 5 monomers
    top_monomers = monomer_count.most_common(5)

    print("\nTop 5 Most Frequent Monomers:")
    print("=" * 120)
    print(f"{'Rank':<6} {'Name':<40} {'SMILES':<60} {'Count':<10}")
    print("-" * 120)

    for rank, (smiles, count) in enumerate(top_monomers, 1):
        name = monomer_details.get(smiles, 'N/A')
        print(f"{rank:<6} {name:<40} {smiles:<60} {count:<10}")

    print("=" * 120)

    return top_monomers


if __name__ == "__main__":
    show_top_monomers()