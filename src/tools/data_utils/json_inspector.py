"""
JSON Structure Inspector

A utility tool for examining and visualizing the structure of MTG-related JSON files.
Helps with understanding data formats without loading entire large files into memory.

Usage:
    python -m src.tools.data_utils.json_inspector data/cards.json
    python -m src.tools.data_utils.json_inspector data/rules.json

Requirements:
    pip install ijson
"""

import os
import sys
import argparse
from pathlib import Path

# Try to import ijson with a helpful error message if not available
try:
    import ijson
except ImportError:
    print("Error: This tool requires the 'ijson' package.")
    print("Please install it with: pip install ijson")
    print("\nThis is a development utility and not required for core functionality.")
    sys.exit(1)


def print_structure(obj, indent=0, max_depth=5, current_depth=0):
    """
    Recursively print the structure of a Python object with indentation.

    Args:
        obj: The object to inspect
        indent: Current indentation level
        max_depth: Maximum recursion depth to prevent excessive output
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        print(f"{' ' * indent}... (max depth reached)")
        return

    prefix = "  " * indent

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                print(f"{prefix}{key} (object):")
                print_structure(value, indent + 1, max_depth, current_depth + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key} (array):")
                if value:
                    # If first element is a dict, inspect it further
                    if isinstance(value[0], dict):
                        print(f"{prefix}  First element (object):")
                        print_structure(
                            value[0], indent + 2, max_depth, current_depth + 1
                        )
                    elif isinstance(value[0], list):
                        print(f"{prefix}  First element (array):")
                        print_structure(
                            value[0], indent + 2, max_depth, current_depth + 1
                        )
                    else:
                        print(
                            f"{prefix}  Example element type: {type(value[0]).__name__}"
                        )
                        if isinstance(value[0], str) and len(value[0]) < 50:
                            print(f"{prefix}  Example value: '{value[0]}'")
                else:
                    print(f"{prefix}  (empty array)")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")
                # Show short string values
                if isinstance(value, str) and len(value) < 50:
                    print(f"{prefix}  Value: '{value}'")
    elif isinstance(obj, list):
        if obj:
            if len(obj) > 1:
                print(
                    f"{prefix}Array with {len(obj)} items of type {type(obj[0]).__name__}"
                )
            if isinstance(obj[0], (dict, list)):
                print(f"{prefix}First item:")
                print_structure(obj[0], indent + 1, max_depth, current_depth + 1)
            else:
                print(f"{prefix}Example item: {obj[0]}")
        else:
            print(f"{prefix}(empty array)")
    else:
        print(f"{prefix}Value: {obj}")


def inspect_json_structure(filename, max_items=3):
    """
    Inspect the structure of a JSON file using streaming parser.

    Args:
        filename: Path to the JSON file to inspect
        max_items: Maximum number of top-level items to analyze
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return

    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"\nInspecting JSON file: {filename} ({file_size:.2f} MB)")
    print("-" * 80)

    try:
        with open(filename, "rb") as f:
            # First, try to determine top-level structure
            print("Analyzing top-level structure...")
            top_keys = []
            for prefix, event, value in ijson.parse(f):
                if prefix == "" and event == "map_key":
                    top_keys.append(value)
                # Stop after scanning a small portion to save time
                if f.tell() > 100000:  # Read first 100KB
                    break

            if top_keys:
                print(f"Top-level keys: {', '.join(top_keys)}")
            else:
                print("No top-level keys found (possibly an array or invalid JSON)")

            # Reset file position
            f.seek(0)

            # Now try to examine actual content structure
            print("\nExamining content structure:")

            # MTG cards.json typically has 'data' as a top-level key with set codes
            if "data" in top_keys:
                try:
                    data_items = ijson.kvitems(f, "data")
                    for i, (set_code, set_obj) in enumerate(data_items):
                        if i >= max_items:
                            print(
                                f"\nStopping after {max_items} items. File contains more data..."
                            )
                            break
                        print(f"\nSet '{set_code}':")
                        print_structure(set_obj)
                except Exception as e:
                    print(f"Error parsing 'data' section: {e}")

            # MTG rules.json typically has rules in a top-level array
            elif any(k in ["rules", "content"] for k in top_keys):
                key_to_use = next(
                    (k for k in ["rules", "content"] if k in top_keys), top_keys[0]
                )
                try:
                    # Try first as array of objects
                    items = ijson.items(f, f"{key_to_use}.item")
                    for i, item in enumerate(items):
                        if i >= max_items:
                            print(
                                f"\nStopping after {max_items} items. File contains more data..."
                            )
                            break
                        print(f"\nItem {i+1}:")
                        print_structure(item)
                except Exception as e:
                    print(f"Error parsing '{key_to_use}' section: {e}")
                    # Fallback - try as dictionary
                    f.seek(0)
                    try:
                        section = ijson.items(f, key_to_use)
                        obj = next(section, None)
                        if obj:
                            print_structure(obj)
                    except Exception as e2:
                        print(f"Error in fallback parsing: {e2}")

            # Generic approach for other JSON formats
            else:
                try:
                    # Try parsing as a single JSON object
                    parser = ijson.parse(f)
                    current_key = None

                    # Process only first 1000 events to prevent overloading
                    for i, (prefix, event, value) in enumerate(parser):
                        if i > 1000:
                            print(
                                "(Output truncated to prevent excessive processing...)"
                            )
                            break

                        if event == "map_key":
                            current_key = value
                        elif event == "string" and current_key:
                            print(
                                f"Found string value for key '{current_key}': {value[:50]}..."
                            )
                            current_key = None

                        if prefix == "" and event == "end_map":
                            print("Completed scan of top-level object")
                            break

                except Exception as e:
                    print(f"Error parsing JSON structure: {e}")

    except Exception as e:
        print(f"Error opening or processing file: {e}")


def main():
    """Run the JSON inspector command-line tool."""
    parser = argparse.ArgumentParser(description="Inspect the structure of JSON files")
    parser.add_argument("file", help="JSON file to inspect")
    parser.add_argument(
        "--max-items",
        type=int,
        default=3,
        help="Maximum number of top-level items to analyze",
    )

    args = parser.parse_args()
    inspect_json_structure(args.file, args.max_items)


if __name__ == "__main__":
    main()
