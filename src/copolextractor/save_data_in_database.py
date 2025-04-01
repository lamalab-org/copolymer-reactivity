import json
import os
import re
from typing import Dict, List, Any, Union
from urllib.parse import urlparse
from copolextractor.mongodb_storage import CoPolymerDB
import copolextractor.utils as utils


def convert_to_float(value: Any) -> Union[float, Any]:
    """Versucht, einen Wert in Float zu konvertieren; bei Misserfolg wird der Originalwert zurückgegeben."""
    try:
        if isinstance(value, str) and value.lower() == "na":
            return value
        return float(value)
    except (ValueError, TypeError):
        return value


def extract_monomers(monomers_data):
    """
    Extrahiert monomer1 und monomer2 aus den verschiedenen möglichen Formaten.

    Format 1: ["2-hydroxyethyl methacrylate", "acrylamide"]
    Format 2: [{"monomer1": "acryloyl chloride", "monomer2": "2-hydroxypropyl methacrylate"}]
    Format 3: ["monomer1", "2-Hydroxypropyl Methacrylate", "monomer2", "Ethyl Acrylate"]

    Returns:
        tuple: (monomer1, monomer2)
    """
    monomer1 = None
    monomer2 = None

    if not monomers_data:
        return monomer1, monomer2

    # Format 2: Dictionary in einer Liste
    if isinstance(monomers_data, list) and len(monomers_data) > 0 and isinstance(monomers_data[0], dict):
        monomer_dict = monomers_data[0]
        monomer1 = monomer_dict.get("monomer1", None)
        monomer2 = monomer_dict.get("monomer2", None)

    # Format 3: Liste mit "monomer1", "wert1", "monomer2", "wert2"
    elif isinstance(monomers_data, list) and len(monomers_data) >= 4 and monomers_data[
        0] == "monomer1" and "monomer2" in monomers_data:
        monomer2_index = monomers_data.index("monomer2")
        monomer1 = monomers_data[1] if monomer2_index > 1 else None
        monomer2 = monomers_data[monomer2_index + 1] if monomer2_index + 1 < len(monomers_data) else None

    # Format 1: Einfache Liste mit zwei Monomeren
    elif isinstance(monomers_data, list) and len(monomers_data) >= 2:
        monomer1 = monomers_data[0]
        monomer2 = monomers_data[1]

    return monomer1, monomer2


def process_reaction(reaction: Dict, reaction_index: int, source: str, base_filename: str,
                     original_source: str) -> Dict:
    """Process a single reaction entry from the data."""
    # Generate filename with reaction index if needed
    filename = f"{base_filename}_{reaction_index}.json" if reaction_index > 0 else f"{base_filename}.json"

    # Get first reaction condition
    reaction_condition = reaction["reaction_conditions"][0]

    # Konvertiere Temperatur zu Float
    temperature = convert_to_float(reaction_condition["temperature"])

    # Flache Darstellung der Reaktionskonstanten erstellen
    # Fehlerbehandlung für verschiedene Formate der Reaktionskonstanten
    constant_1 = None
    constant_2 = None
    constant_conf_1 = None
    constant_conf_2 = None

    # Überprüfe, ob reaction_constants vorhanden ist und die richtige Struktur hat
    if "reaction_constants" in reaction_condition:
        rc = reaction_condition["reaction_constants"]
        if isinstance(rc, dict):
            constant_1 = convert_to_float(rc.get("constant_1", None))
            constant_2 = convert_to_float(rc.get("constant_2", None))
        else:
            # Fallback für andere Formate
            print(f"Warnung: reaction_constants hat ein unerwartetes Format: {type(rc)}")
            # Versuche, das erste Element als constant_1 und das zweite als constant_2 zu interpretieren, falls es eine Liste ist
            if isinstance(rc, list) and len(rc) >= 2:
                constant_1 = convert_to_float(rc[0])
                constant_2 = convert_to_float(rc[1])

    # Überprüfe, ob reaction_constant_conf vorhanden ist und die richtige Struktur hat
    if "reaction_constant_conf" in reaction_condition:
        rcc = reaction_condition["reaction_constant_conf"]
        if isinstance(rcc, dict):
            constant_conf_1 = rcc.get("constant_conf_1", None)
            constant_conf_2 = rcc.get("constant_conf_2", None)
        else:
            # Fallback für andere Formate
            print(f"Warnung: reaction_constant_conf hat ein unerwartetes Format: {type(rcc)}")
            # Versuche, das erste Element als constant_conf_1 und das zweite als constant_conf_2 zu interpretieren, falls es eine Liste ist
            if isinstance(rcc, list) and len(rcc) >= 2:
                constant_conf_1 = rcc[0]
                constant_conf_2 = rcc[1]

    # Wenn conf-Werte numerisch sind, konvertiere sie zu Float
    constant_conf_1 = convert_to_float(constant_conf_1)
    constant_conf_2 = convert_to_float(constant_conf_2)

    # Build reaction conditions with required fields (flattened structure)
    reaction_conditions = {
        "temperature": temperature,
        "temperature_unit": reaction_condition.get("temperature_unit", "°C"),
        "solvent": reaction_condition.get("solvent", ""),
        "method": reaction_condition.get("method", ""),
        "polymerization_type": reaction_condition.get("polymerization_type", ""),
        "constant_1": constant_1,
        "constant_2": constant_2,
        "constant_conf_1": constant_conf_1,
        "constant_conf_2": constant_conf_2,
        "determination_method": reaction_condition.get("determination_method", "")
    }

    # Berechne das r_product aus constant_1 und constant_2, falls möglich
    if constant_1 is not None and constant_2 is not None and isinstance(constant_1, (int, float)) and isinstance(
            constant_2, (int, float)):
        reaction_conditions["r_product"] = round(constant_1 * constant_2, 4)

    # Add optional fields only if they exist
    if "Q-value" in reaction_condition:
        q_value = reaction_condition["Q-value"]
        if isinstance(q_value, dict):
            reaction_conditions["Q_value_1"] = convert_to_float(q_value.get("constant_1", None))
            reaction_conditions["Q_value_2"] = convert_to_float(q_value.get("constant_2", None))
        else:
            reaction_conditions["Q_value"] = convert_to_float(q_value)

    if "e-Value" in reaction_condition:
        e_value = reaction_condition["e-Value"]
        if isinstance(e_value, dict):
            reaction_conditions["e_value_1"] = convert_to_float(e_value.get("constant_1", None))
            reaction_conditions["e_value_2"] = convert_to_float(e_value.get("constant_2", None))
        else:
            reaction_conditions["e_value"] = convert_to_float(e_value)

    if "r_product" in reaction_condition:
        reaction_conditions["r_product"] = convert_to_float(reaction_condition["r_product"])

    # Extrahiere Monomere aus den verschiedenen möglichen Formaten
    monomer1, monomer2 = extract_monomers(reaction.get("monomers", []))

    # Erstelle eine flache Struktur ohne verschachtelte reaction_conditions
    processed_data = {
        "filename": filename,
        "monomers": reaction["monomers"],  # Original-Format beibehalten
        "monomer1": monomer1,  # Extrahiertes Monomer 1
        "monomer2": monomer2,  # Extrahiertes Monomer 2
        "source": source,
        "original_source": original_source,
        # Füge alle reaction_conditions-Schlüssel direkt in die Hauptstruktur ein
        **reaction_conditions  # Python 3.5+ Dictionary-Unpacking
    }
    return processed_data


def load_data_from_directory(db: CoPolymerDB, directory: str, original_source: str) -> List[Dict]:
    """Load and process all JSON files from a specific directory."""
    results = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return results

    # Process each JSON file
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                print(f"\nProcessing file: {filename}")

                # Load JSON file
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract source and create base filename
                source = data.get("source", "")
                if source and "doi.org/" in source:
                    base_filename = utils.sanitize_filename(source.split("doi.org/")[-1])
                else:
                    base_filename = filename.replace('.json', '')

                print(f"Found source: {source}")
                print(f"Base filename: {base_filename}")

                # Process each reaction
                reactions = data.get("reactions", [])
                print(f"Found {len(reactions)} reactions")

                for i, reaction in enumerate(reactions):
                    try:
                        print(f"Processing reaction {i + 1}/{len(reactions)}")
                        processed_data = process_reaction(
                            reaction=reaction,
                            reaction_index=i,
                            source=source,
                            base_filename=base_filename,
                            original_source=original_source
                        )

                        # Debug-Ausgabe vor dem Speichern
                        print(f"Debug - Processed data structure:")
                        print(f"  Filename: {processed_data['filename']}")
                        print(f"  Monomer1: {processed_data['monomer1']}")
                        print(f"  Monomer2: {processed_data['monomer2']}")
                        print(f"  Toplevel keys: {list(processed_data.keys())}")

                        try:
                            # Save to database
                            result = db.save_data(processed_data)
                            result['filename'] = processed_data['filename']
                            results.append(result)
                            print(f"Processed reaction {i + 1}: {result['message']} "
                                  f"with filename {processed_data['filename']}")
                        except Exception as save_error:
                            print(f"Fehler beim Speichern der Daten: {str(save_error)}")
                            # Detaillierte Fehlermeldung
                            print(f"Fehlerdaten für {processed_data['filename']}:")
                            for key, value in processed_data.items():
                                if key != 'reaction_conditions':
                                    print(f"  {key}: {value}")
                                else:
                                    print(f"  reaction_conditions:")
                                    for rc_key, rc_value in processed_data['reaction_conditions'].items():
                                        print(f"    {rc_key}: {rc_value}")
                            # Füge trotzdem ein Ergebnis hinzu, aber markiere es als Fehler
                            results.append({
                                'success': False,
                                'message': f"Error saving data: {str(save_error)}",
                                'filename': processed_data['filename']
                            })

                    except Exception as e:
                        print(f"Error processing reaction {i + 1} in {filename}: {str(e)}")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    return results


def main(input_path_new_data):
    # Initialize database
    db = CoPolymerDB()

    # Process data from both directories
    print("\nProcessing crossref data...")
    crossref_results = load_data_from_directory(
        db,
        directory=input_path_new_data,
        original_source="crossref"
    )

    print("\nProcessing copol database data...")
    copol_results = load_data_from_directory(
        db,
        directory="../../data_extraction/data_extraction_GPT-4o/output/copol_database/model_output_extraction",
        original_source="copol database"
    )

    # Print summary
    successful_crossref = sum(1 for r in crossref_results if r['success'])
    successful_copol = sum(1 for r in copol_results if r['success'])

    print(f"\nProcessing summary:")
    print(f"Crossref data:")
    print(f"Successfully processed: {successful_crossref}")
    print(f"Failed: {len(crossref_results) - successful_crossref}")

    print(f"\nCopol database data:")
    print(f"Successfully processed: {successful_copol}")
    print(f"Failed: {len(copol_results) - successful_copol}")


if __name__ == "__main__":
    input_path_new_data = "../../data_extraction/model_output_GPT4-o"
    main(input_path_new_data)