from pymongo import MongoClient
from typing import Dict, List, Optional
import datetime


def clean_na_values(data: Dict) -> Dict:
    """
    Convert string NA values to None
    """
    na_strings = ['na', 'n/a', 'null', 'none', 'nan', '']

    for key, value in list(data.items()):
        if isinstance(value, str) and value.lower() in na_strings:
            data[key] = None
        elif isinstance(value, dict):
            data[key] = clean_na_values(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    value[i] = clean_na_values(item)
                elif isinstance(item, str) and item.lower() in na_strings:
                    value[i] = None

    return data


def convert_to_smiles(chemical_name: str) -> Optional[str]:
    """Convert chemical name to SMILES using name_to_smiles"""
    if chemical_name is None:
        return None

    # Skip conversion for objects that aren't strings
    if not isinstance(chemical_name, str):
        print(f"Could not convert '{chemical_name}' to SMILES: not a string")
        return None

    try:
        from copolextractor.utils import name_to_smiles
        smiles = name_to_smiles(chemical_name)
        if smiles:
            print(f"Converted '{chemical_name}' to SMILES: {smiles}")
            return smiles
        else:
            print(f"Could not convert '{chemical_name}' to SMILES.")
            return None
    except Exception as e:
        print(f"Error converting '{chemical_name}' to SMILES: {e}")
        return None


class CoPolymerDB:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", reset_db: bool = False):
        """
        Initialize connection to MongoDB
        """
        self.client = MongoClient(connection_string)
        self.db = self.client.co_polymer_database

        # Drop collections if reset is requested
        if reset_db:
            print("Resetting database: dropping all collections...")
            self.db.co_polymerization_data.drop()
            self.db.monomers.drop()
            print("Database reset complete.")

        # Create or access collections
        self.collection = self.db.co_polymerization_data
        self.monomers_collection = self.db.monomers

        # Create indexes for faster queries
        self.collection.create_index("filename")
        self.collection.create_index([
            ("monomers", 1),
            ("reaction_temperature", 1),
            ("reaction_solvent", 1)
        ])

        # Create indexes for monomers collection
        self.monomers_collection.create_index("name", unique=True)
        self.monomers_collection.create_index("smiles")

    def query_data(self, criteria: Dict = None) -> List[Dict]:
        """
        Query polymer data from database
        """
        if criteria is None:
            criteria = {}

        try:
            return list(self.collection.find(criteria))
        except Exception as e:
            print(f"Error querying data: {str(e)}")
            return []

    def unnest_reaction_data(self, data: Dict) -> Dict:
        """
        Unnest nested structures in the data
        """
        unnested_data = {}

        # Copy top-level fields
        for key, value in data.items():
            if not isinstance(value, dict) and not isinstance(value, list):
                unnested_data[key] = value

        # Process reaction conditions if present
        if "reaction_conditions" in data and isinstance(data["reaction_conditions"], dict):
            reaction_cond = data["reaction_conditions"]

            # Add flattened reaction condition fields
            for key, value in reaction_cond.items():
                if not isinstance(value, dict) and not isinstance(value, list):
                    unnested_data[f"reaction_{key}"] = value

            # Process nested reaction constants if present
            if "reaction_constants" in reaction_cond and isinstance(reaction_cond["reaction_constants"], dict):
                constants = reaction_cond["reaction_constants"]
                for const_key, const_value in constants.items():
                    unnested_data[f"reaction_constant_{const_key}"] = const_value

        # Keep original structure as well
        unnested_data["original_data"] = data

        return unnested_data

    def process_monomers(self, data: Dict) -> Dict:
        """
        Extract monomer information from various formats and convert names to SMILES
        """
        monomers_data = []
        monomers = data.get('monomers', [])
        processed_data = data.copy()

        try:
            # Handle different monomer formats
            if isinstance(monomers, list):
                # Format 1: List of dictionaries with monomer1, monomer2 keys
                if len(monomers) > 0 and isinstance(monomers[0], dict) and (
                        'monomer1' in monomers[0] or 'monomer2' in monomers[0]):
                    for monomer_dict in monomers:
                        for i in range(1, 3):
                            monomer_key = f'monomer{i}'
                            if monomer_key in monomer_dict and monomer_dict[monomer_key]:
                                monomer_name = monomer_dict[monomer_key]
                                if monomer_name not in [None, "None", "none", "NA", "na", ""]:
                                    # Get SMILES from various possible locations
                                    monomer_smiles = data.get(f"monomer{i}_s")
                                    if not monomer_smiles:
                                        monomer_smiles = monomer_dict.get(f"monomer{i}_s")
                                    if not monomer_smiles:
                                        monomer_smiles = convert_to_smiles(monomer_name)

                                    monomers_data.append({
                                        "name": monomer_name,
                                        "smiles": monomer_smiles,
                                        "position": i,
                                        "source_id": data.get("filename", "")
                                    })
                                    processed_data[f"monomer{i}"] = monomer_name
                                    processed_data[f"monomer{i}_s"] = monomer_smiles

                # Format 2: Simple list of monomers
                elif len(monomers) > 0 and all(isinstance(m, str) for m in monomers):
                    for i, monomer_name in enumerate(monomers[:2], 1):
                        if monomer_name and monomer_name not in [None, "None", "none", "NA", "na", ""]:
                            # Get existing SMILES or convert name to SMILES
                            monomer_smiles = data.get(f"monomer{i}_s")
                            if not monomer_smiles:
                                monomer_smiles = convert_to_smiles(monomer_name)

                            monomers_data.append({
                                "name": monomer_name,
                                "smiles": monomer_smiles,
                                "position": i,
                                "source_id": data.get("filename", "")
                            })
                            processed_data[f"monomer{i}"] = monomer_name
                            processed_data[f"monomer{i}_s"] = monomer_smiles

            # Direct monomer1/monomer2 fields
            for i in range(1, 3):
                monomer_name = data.get(f"monomer{i}")
                if monomer_name and isinstance(monomer_name, str) and monomer_name not in ["None", "none", "NA", "na",
                                                                                           ""]:
                    # Get existing SMILES or convert name to SMILES
                    monomer_smiles = data.get(f"monomer{i}_s")
                    if not monomer_smiles:
                        monomer_smiles = convert_to_smiles(monomer_name)

                    # Only add if not already in the list
                    if not any(m.get("name") == monomer_name and m.get("position") == i for m in monomers_data):
                        monomers_data.append({
                            "name": monomer_name,
                            "smiles": monomer_smiles,
                            "position": i,
                            "source_id": data.get("filename", "")
                        })
                        processed_data[f"monomer{i}"] = monomer_name
                        processed_data[f"monomer{i}_s"] = monomer_smiles

            # Process solvent - convert to SMILES if needed
            if "reaction_conditions" in data and isinstance(data["reaction_conditions"], dict):
                solvent = data["reaction_conditions"].get("solvent")
                if solvent and solvent != "bulk" and solvent not in ["None", "none", "NA", "na", ""]:
                    solvent_smiles = convert_to_smiles(solvent)
                    if solvent_smiles:
                        data["reaction_conditions"]["solvent_smiles"] = solvent_smiles

            # For unnested structures
            if "reaction_solvent" in data and data["reaction_solvent"] and data["reaction_solvent"] != "bulk":
                solvent = data["reaction_solvent"]
                if solvent not in ["None", "none", "NA", "na", ""]:
                    solvent_smiles = convert_to_smiles(solvent)
                    if solvent_smiles:
                        data["reaction_solvent_smiles"] = solvent_smiles
                        processed_data["reaction_solvent_smiles"] = solvent_smiles

        except Exception as e:
            print(f"Error processing monomers: {str(e)}")

        return {
            "processed_data": processed_data,
            "monomers_data": monomers_data
        }

    def save_monomers(self, monomers_data: List[Dict]) -> List[str]:
        """
        Save monomers to the monomers collection
        """
        saved_ids = []

        for monomer in monomers_data:
            try:
                # Only process monomers with valid names
                if not monomer.get("name") or monomer.get("name") in ["None", "none", "NA", "na", ""]:
                    continue

                # If no SMILES is provided, try to convert the name
                if not monomer.get("smiles"):
                    monomer["smiles"] = convert_to_smiles(monomer["name"])

                # Check if monomer already exists
                existing = self.monomers_collection.find_one({"name": monomer["name"]})

                if existing:
                    # Update smiles if it's now available and wasn't before
                    if monomer.get("smiles") and not existing.get("smiles"):
                        self.monomers_collection.update_one(
                            {"_id": existing["_id"]},
                            {"$set": {"smiles": monomer["smiles"]}}
                        )

                    saved_ids.append(str(existing["_id"]))
                else:
                    # Create new monomer entry
                    new_monomer = {
                        "name": monomer["name"],
                        "smiles": monomer.get("smiles"),
                        "position": monomer.get("position"),
                        "created_at": datetime.datetime.utcnow()
                    }

                    result = self.monomers_collection.insert_one(new_monomer)
                    saved_ids.append(str(result.inserted_id))

            except Exception as e:
                print(f"Error saving monomer {monomer.get('name')}: {str(e)}")

        return saved_ids

    def check_duplicate(self, data: Dict) -> Optional[Dict]:
        """
        Check if entry already exists with same source and same content.
        If the indices match but content differs, suggest a new index.
        """
        # Base query - look for same source and indices
        base_query = {}
        if "source" in data:
            base_query["source"] = data["source"]
        if "reaction_index" in data:
            base_query["reaction_index"] = data["reaction_index"]
        if "condition_index" in data:
            base_query["condition_index"] = data["condition_index"]

        # If we don't have base fields, use filename as fallback
        if not base_query and "filename" in data:
            base_query["filename"] = data["filename"]

        # If we still don't have a query, we can't check for duplicates
        if not base_query:
            return None

        # Find entries with matching base criteria
        matching_entries = list(self.collection.find(base_query))

        # If no matches, definitely not a duplicate
        if not matching_entries:
            return None

        # Check if any of the matching entries have the same content
        for entry in matching_entries:
            # Check if monomers match
            monomers_match = True
            if "monomers" in data and "monomers" in entry:
                if data["monomers"] != entry["monomers"]:
                    monomers_match = False

            # Check if reaction conditions match
            conditions_match = True
            # For unnested data
            if all(k.startswith("reaction_") for k in data.keys() if k.startswith("reaction_")):
                reaction_keys = [k for k in data.keys() if k.startswith("reaction_")]
                for key in reaction_keys:
                    if key in data and key in entry and data[key] != entry[key]:
                        conditions_match = False
                        break
            # For nested data
            elif "reaction_conditions" in data and "reaction_conditions" in entry:
                if data["reaction_conditions"] != entry["reaction_conditions"]:
                    conditions_match = False

            # If both monomers and conditions match, this is a true duplicate
            if monomers_match and conditions_match:
                return entry

        # If we get here, we found entries with matching indices but different content
        # Let's find the next available indices
        if "condition_index" in data:
            # Find the highest condition_index for this reaction
            max_condition_index = max([e.get("condition_index", -1) for e in matching_entries])
            data["condition_index"] = max_condition_index + 1
            print(f"Updating condition_index to {data['condition_index']} for different content")
        elif "reaction_index" in data:
            # Find the highest reaction_index for this source
            all_source_entries = list(self.collection.find({"source": data.get("source")}))
            max_reaction_index = max([e.get("reaction_index", -1) for e in all_source_entries])
            data["reaction_index"] = max_reaction_index + 1
            data["condition_index"] = 0
            print(f"Updating reaction_index to {data['reaction_index']} for different content")

        # This isn't a duplicate, but we've updated the indices
        return None

    def save_data(self, data: Dict, check_duplicates: bool = True, unnest: bool = True) -> Dict:
        """
        Save new polymer data to database
        """
        try:
            # First, clean NA values in the data
            data = clean_na_values(data)

            if check_duplicates:
                existing = self.check_duplicate(data)
                if existing:
                    return {
                        "success": False,
                        "message": "Duplicate entry found",
                        "existing_data": existing
                    }

            # Ensure filename is present
            if "filename" not in data:
                # Generate a filename if not present
                source = data.get("source", "unknown")
                source_id = source.split("/")[-1] if "/" in source else source
                reaction_idx = data.get("reaction_index", 0)
                condition_idx = data.get("condition_index", 0)
                data["filename"] = f"{source_id}_r{reaction_idx}_c{condition_idx}.json"

            # Add timestamp
            data["created_at"] = datetime.datetime.utcnow()

            # Process the data
            if unnest:
                # Process monomers
                processed_result = self.process_monomers(data)
                processed_data = processed_result["processed_data"]
                monomers_data = processed_result["monomers_data"]

                # Unnest the data structure
                unnested_data = self.unnest_reaction_data(processed_data)

                # Save monomers to separate collection
                monomer_ids = self.save_monomers(monomers_data)
                unnested_data["monomer_references"] = monomer_ids

                # Save the unnested data
                result = self.collection.insert_one(unnested_data)
            else:
                # Save original data without unnesting
                result = self.collection.insert_one(data)

            return {
                "success": True,
                "message": "Data saved successfully",
                "id": str(result.inserted_id)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error saving data: {str(e)}"
            }

    def process_json_file(self, json_data: Dict) -> List[Dict]:
        """
        Process a JSON file with multiple reactions
        """
        # First, clean NA values in the entire JSON data
        json_data = clean_na_values(json_data)

        entries = []

        # Extract source ID for filenames
        source_id = ""
        if "source" in json_data and json_data["source"]:
            source_id = json_data["source"].split("/")[-1]
        elif "filename" in json_data and json_data["filename"]:
            source_id = json_data["filename"].split(".")[0]

        # Extract common metadata
        metadata = {
            "source": json_data.get("source"),
            "PDF_name": json_data.get("PDF_name"),
            "original_filename": json_data.get("filename", "")
        }

        # Process reactions
        if "reactions" in json_data and isinstance(json_data["reactions"], list):
            for reaction_idx, reaction in enumerate(json_data["reactions"]):
                # Get monomers
                monomers = reaction.get("monomers", [])

                # Process each reaction condition
                if "reaction_conditions" in reaction and isinstance(reaction["reaction_conditions"], list):
                    for cond_idx, condition in enumerate(reaction["reaction_conditions"]):
                        # Create a new entry for each combination
                        entry = metadata.copy()

                        # Add monomers
                        entry["monomers"] = monomers

                        # Add specific monomer fields
                        if len(monomers) >= 2:
                            entry["monomer1"] = monomers[0]
                            entry["monomer2"] = monomers[1]

                        # Add reaction condition
                        entry["reaction_conditions"] = condition

                        # Add indices for reference
                        entry["reaction_index"] = reaction_idx
                        entry["condition_index"] = cond_idx

                        # Generate a unique filename
                        entry["filename"] = f"{source_id}_r{reaction_idx}_c{cond_idx}.json"

                        entries.append(entry)

                # If there's only one reaction condition directly
                elif not "reaction_conditions" in reaction and isinstance(reaction, dict):
                    # Extract reaction conditions directly
                    condition = {k: v for k, v in reaction.items()
                                 if k != "monomers" and not k.startswith("monomer")}

                    if condition:
                        entry = metadata.copy()
                        entry["monomers"] = monomers

                        if len(monomers) >= 2:
                            entry["monomer1"] = monomers[0]
                            entry["monomer2"] = monomers[1]

                        entry["reaction_conditions"] = condition
                        entry["reaction_index"] = reaction_idx
                        entry["condition_index"] = 0

                        # Generate a unique filename
                        entry["filename"] = f"{source_id}_r{reaction_idx}.json"

                        entries.append(entry)

        # If no explicit reactions, treat as one reaction
        elif "monomers" in json_data:
            monomers = json_data.get("monomers", [])

            # Check if we have reaction_conditions as a list
            if "reaction_conditions" in json_data and isinstance(json_data["reaction_conditions"], list):
                for cond_idx, condition in enumerate(json_data["reaction_conditions"]):
                    entry = metadata.copy()
                    entry["monomers"] = monomers

                    if len(monomers) >= 2:
                        entry["monomer1"] = monomers[0]
                        entry["monomer2"] = monomers[1]

                    entry["reaction_conditions"] = condition
                    entry["condition_index"] = cond_idx

                    # Generate a unique filename
                    entry["filename"] = f"{source_id}_c{cond_idx}.json"

                    entries.append(entry)
            else:
                # Extract reaction conditions directly
                condition = {k: v for k, v in json_data.items()
                             if k not in ["source", "PDF_name", "filename", "monomers"]
                             and not k.startswith("monomer")}

                if condition:
                    entry = metadata.copy()
                    entry["monomers"] = monomers

                    if len(monomers) >= 2:
                        entry["monomer1"] = monomers[0]
                        entry["monomer2"] = monomers[1]

                    entry["reaction_conditions"] = condition

                    # Generate a unique filename
                    entry["filename"] = f"{source_id}.json"

                    entries.append(entry)

        return entries

    def save_json_file(self, json_data: Dict, check_duplicates: bool = True) -> Dict:
        """
        Process and save a JSON file with possibly multiple reactions
        """
        # Process the JSON file into individual entries
        entries = self.process_json_file(json_data)

        if not entries:
            return {
                "success": False,
                "message": "No valid reaction entries found in the JSON data"
            }

        results = {
            "success": True,
            "total_entries": len(entries),
            "saved_entries": 0,
            "duplicate_entries": 0,
            "failed_entries": 0,
            "entry_results": []
        }

        # Save each entry
        for entry in entries:
            try:
                save_result = self.save_data(entry, check_duplicates, unnest=True)
                results["entry_results"].append(save_result)

                if save_result["success"]:
                    results["saved_entries"] += 1
                elif "Duplicate entry found" in save_result.get("message", ""):
                    results["duplicate_entries"] += 1
                else:
                    results["failed_entries"] += 1
                    print(f"Failed to save entry: {save_result.get('message')}")
            except Exception as e:
                results["failed_entries"] += 1
                print(f"Exception while saving entry: {str(e)}")
                results["entry_results"].append({
                    "success": False,
                    "message": f"Error processing entry: {str(e)}"
                })

        # Update overall success status
        results["success"] = results["saved_entries"] > 0
        results[
            "message"] = f"Processed {results['total_entries']} entries: {results['saved_entries']} saved, {results['duplicate_entries']} duplicates, {results['failed_entries']} failed"

        return results


# Simple usage example
if __name__ == "__main__":
    # Initialize database
    db = CoPolymerDB()

    # Example JSON with multiple reaction conditions
    example_json = {
        "reactions": [
            {
                "monomers": ["benzyl methacrylate", "diethylaminoethyl methacrylate"],
                "reaction_conditions": [
                    {
                        "polymerization_type": "free radical",
                        "solvent": "benzene",
                        "temperature": 60.0,
                        "temperature_unit": "°C",
                        "determination_method": "Fineman-Ross",
                        "reaction_constants": {
                            "constant_1": 0.588,
                            "constant_2": 0.393
                        }
                    },
                    {
                        "polymerization_type": "free radical",
                        "solvent": "benzene",
                        "temperature": 60.0,
                        "temperature_unit": "°C",
                        "determination_method": "inverted Fineman-Ross",
                        "reaction_constants": {
                            "constant_1": 0.627,
                            "constant_2": 0.43
                        }
                    }
                ]
            }
        ],
        "source": "https://doi.org/10.21767/2472-1123.100006",
        "PDF_name": "Statistical Copolymers Example"
    }

    # Save to database
    result = db.save_json_file(example_json)
    print(result["message"])

    # Query data
    data = db.query_data({})
    print(f"Found {len(data)} database entries")

    # Query monomers
    monomers = db.query_monomers({})
    print(f"Found {len(monomers)} unique monomers")