import os
import json
import csv
import pandas as pd
from tqdm import tqdm
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB


def calculate_missing_properties(smiles, existing_properties):
    """Calculate only missing properties for a molecule."""
    missing_properties = {}

    if (
            "best_conformer_elements" in existing_properties
            and "best_conformer_coordinates" in existing_properties
    ):
        elements = existing_properties["best_conformer_elements"]
        coordinates = existing_properties["best_conformer_coordinates"]
    else:
        print(f"Best conformer data is missing for {smiles}. Calculating it.")
        elements, coordinates, energy = conformer_opt(smiles)
        missing_properties.update(
            {
                "best_conformer_elements": elements,
                "best_conformer_coordinates": coordinates,
                "best_conformer_energy": energy,
            }
        )

    xtb = XTB(elements, coordinates)

    # Calculate and store missing properties
    xtb_properties = [
        ("ip", xtb.get_ip),
        ("ip_corrected", lambda: xtb.get_ip(corrected=True)),
        ("ea", xtb.get_ea),
        ("homo", xtb.get_homo),
        ("lumo", xtb.get_lumo),
        ("charges", xtb.get_charges),
        ("dipole", lambda: xtb.get_dipole().tolist()),
        (
            "global_electrophilicity",
            lambda: xtb.get_global_descriptor("electrophilicity", corrected=True),
        ),
        (
            "global_nucleophilicity",
            lambda: xtb.get_global_descriptor("nucleophilicity", corrected=True),
        ),
        ("fukui_electrophilicity", lambda: xtb.get_fukui("electrophilicity")),
        ("fukui_nucleophilicity", lambda: xtb.get_fukui("nucleophilicity")),
        ("fukui_radical", lambda: xtb.get_fukui("radical")),
    ]

    for prop, func in xtb_properties:
        if prop not in existing_properties:
            try:
                missing_properties[prop] = func()
            except Exception as e:
                print(f"Error calculating {prop} for {smiles}: {e}")

    return missing_properties


def conformer_opt(smiles):
    """Optimize conformer for a given SMILES."""
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()

    try:
        print("Started GFN-FF optimization.")
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN-FF"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()

    try:
        print("Started GFN2-xTB optimization.")
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN2-xTB"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN2-xTB optimization failed for {smiles}: {e}")

    ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
    best_conformer = ce.conformers[0]

    elements = best_conformer.elements.tolist()
    coordinates = best_conformer.coordinates.tolist()
    energy = best_conformer.energy

    return elements, coordinates, energy


def calculate_property(smiles):
    """Calculate all properties for a given SMILES."""
    elements, coordinates, energy = conformer_opt(smiles)
    xtb = XTB(elements, coordinates)

    properties = {
        "smiles": smiles,
        "best_conformer_coordinates": coordinates,
        "best_conformer_elements": elements,
        "best_conformer_energy": energy,
        "ip": xtb.get_ip(),
        "ip_corrected": xtb.get_ip(corrected=True),
        "ea": xtb.get_ea(),
        "homo": xtb.get_homo(),
        "lumo": xtb.get_lumo(),
        "charges": xtb.get_charges(),
        "dipole": xtb.get_dipole().tolist(),
        "global_electrophilicity": xtb.get_global_descriptor(
            "electrophilicity", corrected=True
        ),
        "global_nucleophilicity": xtb.get_global_descriptor(
            "nucleophilicity", corrected=True
        ),
        "fukui_electrophilicity": xtb.get_fukui("electrophilicity"),
        "fukui_nucleophilicity": xtb.get_fukui("nucleophilicity"),
        "fukui_radical": xtb.get_fukui("radical"),
    }

    return properties


def save_properties(properties, filename):
    """Save properties to a file."""
    with open(filename, "w") as f:
        json.dump(properties, f, indent=4)


def load_smiles_from_csv(csv_path):
    """
    Load monomer SMILES from a CSV file.
    """
    unique_smiles = set()

    try:
        # Try using pandas for better handling of potential CSV issues
        df = pd.read_csv(csv_path)

        # Check if the expected columns exist
        if 'monomer1_smiles' in df.columns:
            monomer1_smiles = df['monomer1_smiles'].dropna().unique()
            unique_smiles.update(monomer1_smiles)
            print(f"Found {len(monomer1_smiles)} unique monomer1 SMILES")

        if 'monomer2_smiles' in df.columns:
            monomer2_smiles = df['monomer2_smiles'].dropna().unique()
            unique_smiles.update(monomer2_smiles)
            print(f"Found {len(monomer2_smiles)} unique monomer2 SMILES")

        print(f"Total of {len(unique_smiles)} unique SMILES strings loaded from CSV")

    except Exception as e:
        print(f"Error loading CSV file with pandas: {e}")
        print("Trying alternative CSV parsing method...")

        # Fallback to standard CSV parsing
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    if 'monomer1_smiles' in row and row['monomer1_smiles']:
                        unique_smiles.add(row['monomer1_smiles'])
                    if 'monomer2_smiles' in row and row['monomer2_smiles']:
                        unique_smiles.add(row['monomer2_smiles'])

            print(f"Loaded {len(unique_smiles)} unique SMILES with fallback method")

        except Exception as e2:
            print(f"Both CSV parsing methods failed: {e2}")
            raise

    return unique_smiles


def main():
    """
    Main function that loads SMILES from CSV and calculates properties
    """
    # Fixed parameters
    csv_path = "../data_extraction/extracted_reactions.csv"
    output_folder = "output/molecule_properties"
    smiles_error_path = "output/smiles_error.json"

    # Load or initialize error log
    try:
        with open(smiles_error_path, "r") as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    # Extract unique SMILES from CSV file
    unique_smiles = load_smiles_from_csv(csv_path)

    if not unique_smiles:
        print("No SMILES found in CSV file. Please check the file format.")
        return

    print(f"Found {len(unique_smiles)} unique SMILES strings")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Also ensure the output directory for error log exists
    os.makedirs(os.path.dirname(smiles_error_path), exist_ok=True)

    successful_calculations = 0
    skipped_calculations = 0
    error_calculations = 0

    # Process each unique SMILES
    for smiles in tqdm(sorted(unique_smiles), desc="Processing SMILES"):
        print(f"\nProcessing {smiles}")
        # Create a safe filename by replacing problematic characters
        safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = os.path.join(output_folder, f"{safe_smiles}.json")

        if os.path.exists(filename):
            print(f"Loading existing file: {filename}")
            try:
                # Load existing properties and check if file is empty
                with open(filename, "r") as f:
                    file_content = f.read()
                    if not file_content.strip():
                        print(f"Empty file found for {smiles}, calculating properties")
                        raise ValueError("Empty file")

                    existing_properties = json.loads(file_content)

                # Calculate any missing properties
                missing_properties = calculate_missing_properties(smiles, existing_properties)
                if missing_properties:
                    print(f"Adding missing properties for {smiles}")
                    existing_properties.update(missing_properties)
                    save_properties(existing_properties, filename)
                    successful_calculations += 1
                else:
                    print(f"All properties already exist for {smiles}")
                    skipped_calculations += 1

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading properties for {smiles}: {e}")
                try:
                    # Calculate all properties if file is empty or corrupt
                    print(f"Recalculating all properties for {smiles}")
                    properties = calculate_property(smiles)
                    save_properties(properties, filename)
                    successful_calculations += 1
                except Exception as calc_error:
                    print(f"Error calculating properties for {smiles}: {calc_error}")
                    smiles_error.append({"smiles": smiles, "error": str(calc_error)})
                    with open(smiles_error_path, "w") as f:
                        json.dump(smiles_error, f, indent=4)
                    error_calculations += 1

        elif smiles not in [error["smiles"] for error in smiles_error]:
            try:
                # Calculate all properties for new SMILES
                print(f"Calculating new properties for {smiles}")
                properties = calculate_property(smiles)
                save_properties(properties, filename)
                successful_calculations += 1
            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                smiles_error.append({"smiles": smiles, "error": str(e)})
                with open(smiles_error_path, "w") as f:
                    json.dump(smiles_error, f, indent=4)
                error_calculations += 1
        else:
            print(f"Skipping {smiles} - previous calculation error recorded")
            skipped_calculations += 1

    # Save summary of results
    summary = {
        "total_unique_smiles": len(unique_smiles),
        "successful_calculations": successful_calculations,
        "skipped_calculations": skipped_calculations,
        "error_calculations": error_calculations,
        "error_details": smiles_error
    }

    with open(os.path.join(output_folder, "calculation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nCalculation Summary:")
    print(f"Total unique SMILES: {len(unique_smiles)}")
    print(f"Successfully calculated: {successful_calculations}")
    print(f"Skipped (already exists): {skipped_calculations}")
    print(f"Calculation errors: {error_calculations}")
    print(f"Results saved to {output_folder}")
    print(f"Error log saved to {smiles_error_path}")


if __name__ == "__main__":
    main()