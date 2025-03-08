import ijson


def print_structure(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                print(f"{prefix}{key} (object):")
                print_structure(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}{key} (array):")
                if value:
                    # if first element is a dict, inspect it further
                    if isinstance(value[0], dict):
                        print(f"{prefix}  First element (object):")
                        print_structure(value[0], indent + 2)
                    elif isinstance(value[0], list):
                        print(f"{prefix}  First element (array):")
                        print_structure(value[0], indent + 2)
                    else:
                        print(
                            f"{prefix}  Example element type: {type(value[0]).__name__}"
                        )
                else:
                    print(f"{prefix}  (empty array)")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")


def inspect_json_structure(filename):
    with open(filename, "rb") as f:
        # First, get meta keys (if needed)
        meta_keys = []
        for prefix, event, value in ijson.parse(f):
            if prefix == "meta" and event == "map_key":
                meta_keys.append(value)
            if prefix == "data":
                break
        print("Meta keys:", meta_keys)

        # Reset pointer and get first set in 'data'
        f.seek(0)
        data_items = ijson.kvitems(f, "data")
        try:
            set_code, set_obj = next(data_items)
            print(
                "\nInspecting structure of first set in 'data' (code '{}'):\n".format(
                    set_code
                )
            )
            print_structure(set_obj)
        except StopIteration:
            print("No sets found in 'data'.")


if __name__ == "__main__":
    inspect_json_structure("cards.json")
