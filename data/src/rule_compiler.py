#!/usr/bin/env python3
"""
MTG Rules Text Extractor

This script parses Magic: The Gathering rules text from an input file and
outputs two JSON files: one for the rules hierarchy and one for the glossary.

Usage:
    python parse_mtgrules.py -i INPUT_FILE [-r RULES_OUTPUT] [-g GLOSSARY_OUTPUT]

Arguments:
    -i, --input         Path to the input text file containing the rules.
    -r, --rules         Name of the output file for the rules JSON (default: rules.json).
    -g, --glossary      Name of the output file for the glossary JSON (default: glossary.json).

Example:
    python parse_mtgrules.py -i mtg_rules.txt -r mtg_rules.json -g mtg_glossary.json
"""

import re
import json
import sys
import argparse


def parse_rule_number(rule_str):
    """
    Converts a rule number string (e.g. "100", "100.1", "100.1a")
    into a tuple for hierarchical comparison.

    Args:
        rule_str (str): The rule number string.

    Returns:
        tuple: A tuple representing the hierarchical components of the rule.
    """
    parts = rule_str.split(".")
    result = []
    for part in parts:
        m = re.match(r"(\d+)([a-z]*)", part)
        if m:
            result.append(int(m.group(1)))
            if m.group(2):
                result.append(m.group(2))
        else:
            result.append(part)
    return tuple(result)


def process_embedded_subrules(rule):
    """
    Recursively processes a rule to extract embedded subrules from its text.

    Args:
        rule (dict): A rule dictionary with keys 'text' and 'subrules'.
    """
    main_text, embedded = split_embedded_subrules(rule["text"])
    rule["text"] = main_text
    rule["subrules"].extend(embedded)
    for sub in rule["subrules"]:
        process_embedded_subrules(sub)


def insert_rule(rule_stack, current_section, new_rule, comp):
    """
    Inserts a new rule into the current rule_stack or section based on its hierarchical position.

    Args:
        rule_stack (list): Stack of tuples (comp, rule) representing current hierarchy.
        current_section (dict): The current section dictionary.
        new_rule (dict): The new rule to be inserted.
        comp (tuple): The parsed representation of the new rule number.

    Returns:
        tuple: (updated rule_stack, updated current_section)
    """
    if not rule_stack:
        # Create a new section if none exists
        if current_section is None:
            current_section = {"section_number": "", "title": "", "rules": []}
        current_section["rules"].append(new_rule)
        rule_stack.append((comp, new_rule))
        return rule_stack, current_section

    last_comp, last_rule = rule_stack[-1]
    if len(comp) > len(last_comp) and comp[: len(last_comp)] == last_comp:
        # New rule is a subrule of the last rule
        last_rule["subrules"].append(new_rule)
        rule_stack.append((comp, new_rule))
        return rule_stack, current_section

    # Otherwise, find the correct parent by adjusting the stack.
    while rule_stack:
        top_comp, _ = rule_stack[-1]
        if len(comp) == len(top_comp) and comp[:-1] == top_comp[:-1]:
            rule_stack.pop()
            break
        elif len(comp) < len(top_comp) or comp[: len(top_comp)] != top_comp:
            rule_stack.pop()
        else:
            break

    if rule_stack:
        parent_rule = rule_stack[-1][1]
        parent_rule["subrules"].append(new_rule)
    else:
        current_section["rules"].append(new_rule)
    rule_stack.append((comp, new_rule))
    return rule_stack, current_section


def split_embedded_subrules(text):
    """
    Splits embedded subrules from the text.
    Looks for subrule markers (pattern: three digits, dot, one or more digits, then a letter)
    and splits the text accordingly.

    Args:
        text (str): The text to be split.

    Returns:
        tuple: A tuple containing the main text and a list of extracted subrules.
    """
    pattern = re.compile(r"(\d{3}\.\d+[a-z])\s+")
    matches = list(pattern.finditer(text))
    if not matches:
        return text.strip(), []

    main_text = text[: matches[0].start()].strip()
    subrules = []
    for i, match in enumerate(matches):
        rule_num = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sub_text = text[start:end].strip()
        subrules.append({"rule_number": rule_num, "text": sub_text, "subrules": []})
    return main_text, subrules


def parse_structure(text):
    """
    Parses text into a hierarchical structure with three layers:
      - Categories (e.g. "1. Game Concepts")
      - Sections (e.g. "100. General")
      - Rules/Subrules (e.g. "100.1", "100.1a", etc.)

    Returns:
        dict: A dictionary with key "categories" containing the parsed hierarchy.
    """
    lines = text.splitlines()
    schema = {"categories": []}

    # Parser state
    parser_state = {"current_category": None, "current_section": None, "rule_stack": []}

    # Precompile regex patterns for efficiency
    patterns = {
        "category": re.compile(r"^(\d)\.\s+(.+)$"),
        "section": re.compile(r"^(\d{3})\.\s+(.+)$"),
        "rule": re.compile(r"^(\d{3}\.\d+(?:[a-z])?)\.\s+(.*)$"),
    }

    # Process each line
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Try to process the line with each pattern type
        if process_category_line(stripped, patterns["category"], parser_state, schema):
            continue
        elif process_section_line(stripped, patterns["section"], parser_state):
            continue
        elif process_rule_line(stripped, patterns["rule"], parser_state):
            continue
        else:
            # Handle continuation line (additional text for the last rule)
            append_continuation_line(stripped, parser_state)

    # Add any remaining data to the schema
    finalize_schema(parser_state, schema)

    # Process embedded subrules recursively
    process_all_embedded_subrules(schema)

    return schema


def process_category_line(line, pattern, state, schema):
    """Process a potential category line and update the state if matched."""
    match = pattern.match(line)
    if not match:
        return False

    # Flush existing data
    flush_section(state)
    flush_category(state, schema)

    # Create new category
    state["current_category"] = {
        "category_number": match.group(1),
        "title": match.group(2).strip(),
        "sections": [],
    }
    return True


def process_section_line(line, pattern, state):
    """Process a potential section line and update the state if matched."""
    match = pattern.match(line)
    if not match:
        return False

    # Flush existing section
    flush_section(state)

    # Create new section
    state["current_section"] = {
        "section_number": match.group(1),
        "title": match.group(2).strip(),
        "rules": [],
    }
    state["rule_stack"] = []
    return True


def process_rule_line(line, pattern, state):
    """Process a potential rule line and update the state if matched."""
    match = pattern.match(line)
    if not match:
        return False

    rule_num_str = match.group(1)
    rule_text = match.group(2).strip()
    new_rule = {"rule_number": rule_num_str, "text": rule_text, "subrules": []}

    comp = parse_rule_number(rule_num_str)
    state["rule_stack"], state["current_section"] = insert_rule(
        state["rule_stack"], state["current_section"], new_rule, comp
    )
    return True


def append_continuation_line(line, state):
    """Append text to the most recently processed rule."""
    if state["rule_stack"]:
        state["rule_stack"][-1][1]["text"] += " " + line
    elif state["current_section"] and state["current_section"]["rules"]:
        state["current_section"]["rules"][-1]["text"] += " " + line


def flush_section(state):
    """Flush the current section into the current category."""
    if state["current_section"] and state["current_category"]:
        state["current_category"]["sections"].append(state["current_section"])
        state["current_section"] = None
        state["rule_stack"] = []


def flush_category(state, schema):
    """Flush the current category into the schema."""
    if state["current_category"]:
        schema["categories"].append(state["current_category"])
        state["current_category"] = None


def finalize_schema(state, schema):
    """Add any remaining data in the state to the schema."""
    if state["current_section"] and state["current_category"]:
        state["current_category"]["sections"].append(state["current_section"])
    if state["current_category"]:
        schema["categories"].append(state["current_category"])


def process_all_embedded_subrules(schema):
    """Process embedded subrules for all rules in the schema."""
    for cat in schema["categories"]:
        for sec in cat["sections"]:
            for rule in sec["rules"]:
                process_embedded_subrules(rule)


def extract_related_rules(definition):
    """
    Extracts rule numbers from the definition.
    A rule number is defined as three digits optionally followed by a dot,
    one or more digits, and an optional letter (e.g., "115.4" or "100.1a").

    Args:
        definition (str): The text definition to search.

    Returns:
        list: A list of the matching rule numbers.
    """
    pattern = r"\b\d{3}(?:\.\d+(?:[a-z])?)?\b"
    matches = re.findall(pattern, definition)
    return matches


def parse_glossary(text):
    """
    Parses glossary text into a list of glossary entries.
    Assumes that each glossary entry is separated by one or more blank lines.
    In each block, the first line is the term and the remaining lines (if any)
    form the definition.

    Args:
        text (str): The glossary text to parse.

    Returns:
        list: A list of glossary entry dictionaries.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    glossary = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        term = lines[0]
        definition = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
        related = extract_related_rules(definition)
        glossary.append(
            {"term": term, "definition": definition, "related_rules": related}
        )
    return glossary


def parse_file(input_file):
    """
    Reads and parses the input file to extract the rules structure and glossary entries.

    Args:
        input_file (str): Path to the input file.

    Returns:
        tuple: (rules_schema, glossary_entries)
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()
    except IOError as e:
        sys.exit(f"Error reading input file: {e}")

    # Assumes that detailed rules appear after the "Credits" marker and that the glossary
    # section appears later (after a "Glossary" marker).
    credits_match = re.search(r"(?im)^\s*Credits\s*$", full_text)
    if credits_match:
        rules_text = full_text[credits_match.end() :]
        glossary_match = re.search(r"(?im)^\s*Glossary\s*$", rules_text)
        if glossary_match:
            glossary_text = rules_text[glossary_match.end() :]
            rules_text = rules_text[: glossary_match.start()]
        else:
            glossary_text = ""
    else:
        rules_text = full_text
        glossary_text = ""

    rules_schema = parse_structure(rules_text)
    glossary_entries = parse_glossary(glossary_text)

    return rules_schema, glossary_entries


def main():
    """
    Main function to handle command-line arguments and initiate parsing.
    """
    parser = argparse.ArgumentParser(
        description="Extract MTG rules text into JSON files for rules and glossary."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input text file containing MTG rules.",
    )
    parser.add_argument(
        "-r",
        "--rules",
        default="rules.json",
        help="Name of the output file for the rules JSON (default: rules.json).",
    )
    parser.add_argument(
        "-g",
        "--glossary",
        default="glossary.json",
        help="Name of the output file for the glossary JSON (default: glossary.json).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="MTG Rules Extractor 1.0",
        help="Display the version of the script and exit.",
    )
    args = parser.parse_args()

    rules_schema, glossary_entries = parse_file(args.input)

    try:
        with open(args.rules, "w", encoding="utf-8") as f:
            json.dump(rules_schema, f, indent=2, ensure_ascii=False)
        with open(args.glossary, "w", encoding="utf-8") as f:
            json.dump(glossary_entries, f, indent=2, ensure_ascii=False)
    except IOError as e:
        sys.exit(f"Error writing output files: {e}")

    print(
        f"Parsing complete. Rules written to '{args.rules}' and glossary written to '{args.glossary}'."
    )


if __name__ == "__main__":
    main()
