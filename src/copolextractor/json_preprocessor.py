#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from mongodb_storage import CoPolymerDB


def process_directory(directory_path, reset_db=False):
    """
    Process all JSON files in a directory and import them into MongoDB

    Args:
        directory_path: Path to directory containing JSON files
        reset_db: Whether to reset the database before processing
    """
    # Initialize the database
    db = CoPolymerDB(reset_db=reset_db)

    if reset_db:
        print("Database has been reset.")

    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in directory.")

    stats = {
        "processed_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "total_entries": 0,
        "saved_entries": 0,
        "duplicate_entries": 0,
        "failed_entries": 0
    }

    # Process each file
    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)
        print(f"\nProcessing file: {file_name}")

        try:
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Add filename to the data
            json_data["filename"] = file_name

            # Save to database
            result = db.save_json_file(json_data)

            # Update statistics
            stats["processed_files"] += 1
            if result["success"]:
                stats["successful_files"] += 1
            else:
                stats["failed_files"] += 1

            stats["total_entries"] += result.get("total_entries", 0)
            stats["saved_entries"] += result.get("saved_entries", 0)
            stats["duplicate_entries"] += result.get("duplicate_entries", 0)
            stats["failed_entries"] += result.get("failed_entries", 0)

            # Print results
            print(f"Result: {result['message']}")

        except Exception as e:
            stats["failed_files"] += 1
            stats["processed_files"] += 1
            print(f"Error processing file {file_name}: {str(e)}")

    # Print summary
    print("\n===== Processing Summary =====")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Successful files: {stats['successful_files']}")
    print(f"Failed files: {stats['failed_files']}")
    print(f"Total entries found: {stats['total_entries']}")
    print(f"Entries saved: {stats['saved_entries']}")
    print(f"Duplicate entries: {stats['duplicate_entries']}")
    print(f"Failed entries: {stats['failed_entries']}")


def process_file(file_path, reset_db=False):
    """
    Process a single JSON file and import it into MongoDB

    Args:
        file_path: Path to the JSON file
        reset_db: Whether to reset the database before processing
    """
    # Initialize the database
    db = CoPolymerDB(reset_db=reset_db)

    if reset_db:
        print("Database has been reset.")

    print(f"Processing file: {file_path}")

    try:
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Add filename to the data
        file_name = os.path.basename(file_path)
        json_data["filename"] = file_name

        # Save to database
        result = db.save_json_file(json_data)

        # Print results
        print(f"\nResult: {result['message']}")
        print(f"Entries saved: {result.get('saved_entries', 0)}")
        print(f"Duplicate entries: {result.get('duplicate_entries', 0)}")
        print(f"Failed entries: {result.get('failed_entries', 0)}")

        # Show details for any failed entries
        if result.get('failed_entries', 0) > 0:
            print("\nFailed entries details:")
            for entry_result in result.get('entry_results', []):
                if not entry_result.get('success'):
                    print(f"- {entry_result.get('message')}")

        return result

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return {"success": False, "message": str(e)}


def main(default_path="../../data_extraction/model_output_GPT4-o"):
    """
    Main function to handle command line arguments

    Args:
        default_path: Default path to use if no path is provided via command line
    """
    parser = argparse.ArgumentParser(description='Process JSON files and import them into MongoDB.')
    parser.add_argument('path', nargs='?', default=default_path,
                        help='Path to a JSON file or directory containing JSON files')
    parser.add_argument('--reset', action='store_true', help='Reset the database before processing')

    args = parser.parse_args()

    print(f"Using path: {args.path}")

    if os.path.isdir(args.path):
        process_directory(args.path, args.reset)
    elif os.path.isfile(args.path) and args.path.endswith('.json'):
        process_file(args.path, args.reset)
    else:
        print(f"Error: {args.path} is not a valid JSON file or directory")


if __name__ == "__main__":
    main()