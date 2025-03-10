# src/data/rule_compiler.py
"""
MTG Rules Compiler

This module provides functionality for parsing Magic: The Gathering rules text
and compiling them into structured data for use in the MTG AI Reasoning Assistant.

The module can be used both as a command-line tool and as an imported library:

Command-line usage:
    python -m src.data.rule_compiler -i INPUT_FILE [-r RULES_OUTPUT] [-g GLOSSARY_OUTPUT]

Library usage:
    from src.data.rule_compiler import MTGRuleCompiler

    # Create a compiler instance
    compiler = MTGRuleCompiler()

    # Parse rules from a file
    rules, glossary = compiler.parse_file("mtg_rules.txt")

    # Or parse from text directly
    rules = compiler.parse_structure(rules_text)
    glossary = compiler.parse_glossary(glossary_text)

    # Access the compiled data
    for category in rules["categories"]:
        print(f"Category: {category['title']}")
        # ...
"""

import re
import json
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Match, Pattern

# Configure logging
logger = logging.getLogger(__name__)


class MTGRuleCompiler:
    """
    A compiler for Magic: The Gathering rules text.

    This class provides methods for parsing MTG rules text into structured data,
    including rules hierarchy and glossary entries.
    """

    def __init__(self):
        """Initialize the compiler with precompiled regex patterns."""
        # Precompile regex patterns for efficiency
        self.patterns = {
            "category": re.compile(r"^(\d)\.\s+(.+)$"),
            "section": re.compile(r"^(\d{3})\.\s+(.+)$"),
            "rule": re.compile(r"^(\d{3}\.\d+(?:[a-z])?)\.\s+(.*)$"),
            "rule_number": re.compile(r"\b\d{3}(?:\.\d+(?:[a-z])?)?\b"),
        }

    def parse_rule_number(self, rule_str: str) -> tuple:
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

    def process_embedded_subrules(self, rule: Dict[str, Any]) -> None:
        """
        Recursively processes a rule to extract embedded subrules from its text.

        Args:
            rule (dict): A rule dictionary with keys 'text' and 'subrules'.
        """
        main_text, embedded = self.split_embedded_subrules(rule["text"])
        rule["text"] = main_text
        rule["subrules"].extend(embedded)
        for sub in rule["subrules"]:
            self.process_embedded_subrules(sub)

    def split_embedded_subrules(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
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

    def insert_rule(
        self,
        rule_stack: List[Tuple[tuple, Dict[str, Any]]],
        current_section: Optional[Dict[str, Any]],
        new_rule: Dict[str, Any],
        comp: tuple,
    ) -> Tuple[List[Tuple[tuple, Dict[str, Any]]], Dict[str, Any]]:
        """
        Inserts a new rule into the current rule_stack or section based on its hierarchical position.

        Args:
            rule_stack: Stack of tuples (comp, rule) representing current hierarchy.
            current_section: The current section dictionary.
            new_rule: The new rule to be inserted.
            comp: The parsed representation of the new rule number.

        Returns:
            tuple: (updated rule_stack, updated current_section)
        """
        # Initialize a section if none exists
        if current_section is None:
            current_section = {"section_number": "", "title": "", "rules": []}
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

    def parse_structure(self, text: str) -> Dict[str, Any]:
        """
        Parses text into a hierarchical structure with three layers:
          - Categories (e.g. "1. Game Concepts")
          - Sections (e.g. "100. General")
          - Rules/Subrules (e.g. "100.1", "100.1a", etc.)

        Args:
            text (str): The text to parse.

        Returns:
            dict: A dictionary with key "categories" containing the parsed hierarchy.
        """
        lines = text.splitlines()
        schema = {"categories": []}

        # Parser state
        parser_state = {
            "current_category": None,
            "current_section": None,
            "rule_stack": [],
        }

        # Process each line
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Try to process the line with each pattern type
            if self._process_category_line(stripped, parser_state, schema):
                continue
            elif self._process_section_line(stripped, parser_state):
                continue
            elif self._process_rule_line(stripped, parser_state):
                continue
            else:
                # Handle continuation line (additional text for the last rule)
                self._append_continuation_line(stripped, parser_state)

        # Add any remaining data to the schema
        self._finalize_schema(parser_state, schema)

        # Process embedded subrules recursively
        self._process_all_embedded_subrules(schema)

        return schema

    def _process_category_line(
        self, line: str, state: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """Process a potential category line and update the state if matched."""
        match = self.patterns["category"].match(line)
        if not match:
            return False

        # Flush existing data
        self._flush_section(state)
        self._flush_category(state, schema)

        # Create new category
        state["current_category"] = {
            "category_number": match.group(1),
            "title": match.group(2).strip(),
            "sections": [],
        }
        return True

    def _process_section_line(self, line: str, state: Dict[str, Any]) -> bool:
        """Process a potential section line and update the state if matched."""
        match = self.patterns["section"].match(line)
        if not match:
            return False

        # Flush existing section
        self._flush_section(state)

        # Create new section
        state["current_section"] = {
            "section_number": match.group(1),
            "title": match.group(2).strip(),
            "rules": [],
        }
        state["rule_stack"] = []
        return True

    def _process_rule_line(self, line: str, state: Dict[str, Any]) -> bool:
        """Process a potential rule line and update the state if matched."""
        match = self.patterns["rule"].match(line)
        if not match:
            return False

        rule_num_str = match.group(1)
        rule_text = match.group(2).strip()
        new_rule = {"rule_number": rule_num_str, "text": rule_text, "subrules": []}

        comp = self.parse_rule_number(rule_num_str)
        state["rule_stack"], state["current_section"] = self.insert_rule(
            state["rule_stack"], state["current_section"], new_rule, comp
        )
        return True

    def _append_continuation_line(self, line: str, state: Dict[str, Any]) -> None:
        """Append text to the most recently processed rule."""
        if state["rule_stack"]:
            state["rule_stack"][-1][1]["text"] += " " + line
        elif state["current_section"] and state["current_section"]["rules"]:
            state["current_section"]["rules"][-1]["text"] += " " + line

    def _flush_section(self, state: Dict[str, Any]) -> None:
        """Flush the current section into the current category."""
        if state["current_section"] and state["current_category"]:
            state["current_category"]["sections"].append(state["current_section"])
            state["current_section"] = None
            state["rule_stack"] = []

    def _flush_category(self, state: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Flush the current category into the schema."""
        if state["current_category"]:
            schema["categories"].append(state["current_category"])
            state["current_category"] = None

    def _finalize_schema(self, state: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Add any remaining data in the state to the schema."""
        if state["current_section"] and state["current_category"]:
            state["current_category"]["sections"].append(state["current_section"])
        if state["current_category"]:
            schema["categories"].append(state["current_category"])

    def _process_all_embedded_subrules(self, schema: Dict[str, Any]) -> None:
        """Process embedded subrules for all rules in the schema."""
        for cat in schema["categories"]:
            for sec in cat["sections"]:
                for rule in sec["rules"]:
                    self.process_embedded_subrules(rule)

    def extract_related_rules(self, definition: str) -> List[str]:
        """
        Extracts rule numbers from the definition.
        A rule number is defined as three digits optionally followed by a dot,
        one or more digits, and an optional letter (e.g., "115.4" or "100.1a").

        Args:
            definition (str): The text definition to search.

        Returns:
            list: A list of the matching rule numbers.
        """
        matches = self.patterns["rule_number"].findall(definition)
        return matches

    def parse_glossary(self, text: str) -> List[Dict[str, Any]]:
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
            related = self.extract_related_rules(definition)
            glossary.append(
                {"term": term, "definition": definition, "related_rules": related}
            )
        return glossary

    def parse_file(
        self, input_file: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
            logger.error(f"Error reading input file: {e}")
            raise

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

        rules_schema = self.parse_structure(rules_text)
        glossary_entries = self.parse_glossary(glossary_text)

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
        version="MTG Rules Compiler 1.0",
        help="Display the version of the script and exit.",
    )
    args = parser.parse_args()

    try:
        compiler = MTGRuleCompiler()
        rules_schema, glossary_entries = compiler.parse_file(args.input)

        with open(args.rules, "w", encoding="utf-8") as f:
            json.dump(rules_schema, f, indent=2, ensure_ascii=False)
        with open(args.glossary, "w", encoding="utf-8") as f:
            json.dump(glossary_entries, f, indent=2, ensure_ascii=False)

        print(
            f"Parsing complete. Rules written to '{args.rules}' and glossary written to '{args.glossary}'."
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
