from pymongo import MongoClient
from typing import Dict, List, Optional, Union
import datetime


class CoPolymerDB:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """Initialize connection to MongoDB"""
        self.client = MongoClient(connection_string)
        self.db = self.client.co_polymer_database
        self.collection = self.db.co_polymerization_data

        # Drop old indexes to avoid conflicts
        self.collection.drop_indexes()

        # Create new indexes for faster queries
        self.collection.create_index("filename", unique=True)
        self.collection.create_index([
            ("monomers", 1),
            ("reaction_conditions.temperature", 1),
            ("reaction_conditions.solvent", 1)
        ])

    def check_duplicate(self, data: Dict) -> Optional[Dict]:
        """
        Check if a similar entry already exists by comparing all relevant fields
        """
        query = {
            # All fields must match to be considered a duplicate
            "filename": data["filename"],
            "monomers": data["monomers"],
            "reaction_conditions.temperature": data["reaction_conditions"]["temperature"],
            "reaction_conditions.temperature_unit": data["reaction_conditions"]["temperature_unit"],
            "reaction_conditions.solvent": data["reaction_conditions"]["solvent"],
            "reaction_conditions.method": data["reaction_conditions"]["method"],
            "reaction_conditions.polymerization_type": data["reaction_conditions"]["polymerization_type"],
            "reaction_conditions.reaction_constants.constant_1": data["reaction_conditions"]["reaction_constants"][
                "constant_1"],
            "reaction_conditions.reaction_constants.constant_2": data["reaction_conditions"]["reaction_constants"][
                "constant_2"],
            "reaction_conditions.determination_method": data["reaction_conditions"]["determination_method"],
            "source": data["source"]
        }

        return self.collection.find_one(query)

    def save_data(self, data: Dict, check_duplicates: bool = True) -> Dict:
        """
        Save new polymer data to database
        """
        try:
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
                return {
                    "success": False,
                    "message": "Missing required field: filename"
                }

            # Add timestamp
            data["created_at"] = datetime.datetime.utcnow()

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


# Usage example:
if __name__ == "__main__":
    # Initialize database
    db = CoPolymerDB()

    # Example data structure
    data = {
        "file": "polymer_data_001.json",
        "monomer1_s": "C=C(C)C(=O)OCCO",
        "monomer2_s": "C=CC(N)=O",
        "monomer1": "2-hydroxyethyl methacrylate",
        "monomer2": "acrylamide",
        "r_values": {
            "constant_1": 2.17,
            "constant_2": 0.0
        },
        "conf_intervals": {
            "constant_conf_1": 0.0,
            "constant_conf_2": 0.0
        },
        "temperature": 60.0,
        "temperature_unit": "°C",
        "solvent": "methanol",
        "solvent_smiles": "CO",
        "logP": -0.3915,
        "method": "bulk",
        "r_product": None,
        "source": "https://doi.org/example",
        "polymerization_type": "free radical",
        "determination_method": "Fineman-Ross"
    }

    # Save data example
    result = db.save_data(data)
    print(result)

    # Query example
    query_result = db.query_data({
        "temperature": 60.0,
        "solvent": "methanol"
    })
    print(query_result)