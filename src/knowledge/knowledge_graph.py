# src/knowledge/knowledge_graph.py
import re
import json
import logging
from datetime import datetime, timezone, UTC
from typing import Dict, List, Set, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class MTGKnowledgeGraph:
    """
    Knowledge graph for Magic: The Gathering entities and relationships.

    This graph maintains structured data about cards, rules, mechanics, and their relationships,
    allowing for complex queries that require graph traversal rather than just semantic similarity.
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.entities = {
            "cards": {},  # card_id -> card_data
            "rules": {},  # rule_id -> rule_data
            "mechanics": {},  # mechanic_id -> mechanic_data
            "keywords": {},  # keyword_id -> keyword_data
        }

        self.relationships = {
            "card_uses_mechanic": [],  # (card_id, mechanic_id)
            "rule_references_rule": [],  # (rule_id, referenced_rule_id)
            "mechanic_governed_by_rule": [],  # (mechanic_id, rule_id)
            "card_referenced_in_rule": [],  # (card_id, rule_id)
            "keyword_appears_on_card": [],  # (keyword_id, card_id)
        }

        # Entity name to ID mapping for quick lookups
        self.name_to_id = {
            "cards": {},
            "mechanics": {},
            "keywords": {},
            "rules": {},
        }

        # Version tracking for cache invalidation
        self.schema_version = "1.0.0"
        self.last_updated = datetime.now(UTC)

        # Statistics tracking
        self.stats = {
            "entity_counts": {entity_type: 0 for entity_type in self.entities},
            "relationship_counts": {rel_type: 0 for rel_type in self.relationships},
            "last_build_time_ms": 0,
        }

    def build_graph_from_data(
        self,
        cards_data: List[Dict],
        rules_data: List[Dict],
        glossary_data: Optional[Dict] = None,
    ) -> None:
        """
        Build the knowledge graph from the provided data sources.

        Args:
            cards_data: List of card data dictionaries
            rules_data: List of rule data dictionaries
            glossary_data: Optional dictionary of MTG glossary terms
        """
        start_time = datetime.now(UTC)

        # Clear existing data
        self._reset_graph()

        # Process cards
        logger.info(f"Processing {len(cards_data)} cards...")
        for card in cards_data:
            if "id" not in card or "name" not in card:
                logger.warning(
                    f"Skipping card with missing id or name: {card.get('name', 'UNKNOWN')}"
                )
                continue

            self._add_card_entity(card)

            # Extract mechanics and keywords from card
            mechanics = self._extract_mechanics(card)
            keywords = self._extract_keywords(card)

            # Add relationships
            for mechanic in mechanics:
                self._add_relationship("card_uses_mechanic", card["id"], mechanic["id"])

            for keyword in keywords:
                self._add_relationship(
                    "keyword_appears_on_card", keyword["id"], card["id"]
                )

        # Process rules
        logger.info(f"Processing {len(rules_data)} rules...")
        for rule in rules_data:
            if "id" not in rule or "text" not in rule:
                logger.warning(
                    f"Skipping rule with missing id or text: {rule.get('id', 'UNKNOWN')}"
                )
                continue

            self._add_rule_entity(rule)

            # Extract rule references from text
            referenced_rules = self._extract_rule_references(rule["text"])
            for ref_rule_id in referenced_rules:
                self._add_relationship("rule_references_rule", rule["id"], ref_rule_id)

            # Extract card references from rules
            referenced_cards = self._extract_card_references(rule["text"])
            for card_id in referenced_cards:
                self._add_relationship("card_referenced_in_rule", card_id, rule["id"])

        # Process glossary data if available
        if glossary_data:
            logger.info("Processing glossary terms...")
            self._process_glossary(glossary_data)

        # Update statistics
        for entity_type in self.entities:
            self.stats["entity_counts"][entity_type] = len(self.entities[entity_type])

        for rel_type in self.relationships:
            self.stats["relationship_counts"][rel_type] = len(
                self.relationships[rel_type]
            )

        # Update version tracking
        self.last_updated = datetime.now(UTC)
        self.stats["last_build_time_ms"] = (
            self.last_updated - start_time
        ).total_seconds() * 1000

        logger.info(
            f"Knowledge graph built in {self.stats['last_build_time_ms']:.2f}ms"
        )
        logger.info(
            f"Entities: {sum(self.stats['entity_counts'].values())}, "
            + f"Relationships: {sum(self.stats['relationship_counts'].values())}"
        )

    def _reset_graph(self) -> None:
        """Reset the graph to an empty state."""
        for entity_type in self.entities:
            self.entities[entity_type] = {}
            self.name_to_id[entity_type] = {}

        for rel_type in self.relationships:
            self.relationships[rel_type] = []

    def _add_card_entity(self, card: Dict[str, Any]) -> None:
        """
        Add a card entity to the graph.

        Args:
            card: Card data dictionary with at least 'id' and 'name' fields
        """
        card_id = card["id"]
        self.entities["cards"][card_id] = card
        self.name_to_id["cards"][card["name"].lower()] = card_id

    def _add_rule_entity(self, rule: Dict[str, Any]) -> None:
        """
        Add a rule entity to the graph.

        Args:
            rule: Rule data dictionary with at least 'id' and 'text' fields
        """
        rule_id = rule["id"]
        self.entities["rules"][rule_id] = rule
        self.name_to_id["rules"][rule_id] = rule_id  # Rules use their ID as "name"

    def _add_mechanic_entity(self, mechanic: Dict[str, Any]) -> None:
        """
        Add a mechanic entity to the graph.

        Args:
            mechanic: Mechanic data dictionary with at least 'id' and 'name' fields
        """
        mechanic_id = mechanic["id"]
        self.entities["mechanics"][mechanic_id] = mechanic
        self.name_to_id["mechanics"][mechanic["name"].lower()] = mechanic_id

    def _add_keyword_entity(self, keyword: Dict[str, Any]) -> None:
        """
        Add a keyword entity to the graph.

        Args:
            keyword: Keyword data dictionary with at least 'id' and 'name' fields
        """
        keyword_id = keyword["id"]
        self.entities["keywords"][keyword_id] = keyword
        self.name_to_id["keywords"][keyword["name"].lower()] = keyword_id

    def _add_relationship(self, relation_type: str, from_id: str, to_id: str) -> None:
        """
        Add a relationship between entities.

        Args:
            relation_type: Type of relationship
            from_id: ID of the source entity
            to_id: ID of the target entity
        """
        if relation_type not in self.relationships:
            logger.warning(f"Unknown relationship type: {relation_type}")
            return

        # Check for duplicates
        relationship = (from_id, to_id)
        if relationship not in self.relationships[relation_type]:
            self.relationships[relation_type].append(relationship)

    def _extract_mechanics(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract mechanics from a card.

        Args:
            card: Card data dictionary

        Returns:
            List of mechanic dictionaries
        """
        mechanics = []

        # Check for mechanics listed directly
        if "mechanics" in card and isinstance(card["mechanics"], list):
            for mechanic_name in card["mechanics"]:
                mechanic_id = f"mechanic_{mechanic_name.lower().replace(' ', '_')}"

                # Create mechanic if it doesn't exist
                if mechanic_id not in self.entities["mechanics"]:
                    mechanic = {
                        "id": mechanic_id,
                        "name": mechanic_name,
                        "source": f"card:{card['id']}",
                    }
                    self._add_mechanic_entity(mechanic)

                mechanics.append({"id": mechanic_id, "name": mechanic_name})

        # Extract mechanics from oracle text
        if "oracle_text" in card and card["oracle_text"]:
            # Look for mechanics in parentheses
            mechanic_pattern = r"\((.*?)\)"
            matches = re.findall(mechanic_pattern, card["oracle_text"])

            for match in matches:
                # Simple heuristic: parenthetical text at the start is often a mechanic
                if match and not any(
                    m["name"].lower() == match.lower() for m in mechanics
                ):
                    mechanic_id = f"mechanic_{match.lower().replace(' ', '_')}"

                    # Create mechanic if it doesn't exist
                    if mechanic_id not in self.entities["mechanics"]:
                        mechanic = {
                            "id": mechanic_id,
                            "name": match,
                            "source": f"card:{card['id']}",
                        }
                        self._add_mechanic_entity(mechanic)

                    mechanics.append({"id": mechanic_id, "name": match})

        return mechanics

    def _extract_keywords(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract keywords from a card.

        Args:
            card: Card data dictionary

        Returns:
            List of keyword dictionaries
        """
        keywords = []

        # Common MTG keywords to look for
        common_keywords = [
            "Flying",
            "First strike",
            "Double strike",
            "Deathtouch",
            "Vigilance",
            "Trample",
            "Haste",
            "Lifelink",
            "Flash",
            "Reach",
            "Defender",
            "Hexproof",
            "Indestructible",
            "Menace",
            "Prowess",
            "Ward",
            "Equip",
        ]

        # Check for keywords in mechanics list first (more reliable)
        if "mechanics" in card and isinstance(card["mechanics"], list):
            for mechanic in card["mechanics"]:
                if mechanic in common_keywords:
                    keyword_id = f"keyword_{mechanic.lower().replace(' ', '_')}"

                    # Create keyword if it doesn't exist
                    if keyword_id not in self.entities["keywords"]:
                        keyword_obj = {
                            "id": keyword_id,
                            "name": mechanic,
                        }
                        self._add_keyword_entity(keyword_obj)

                    keywords.append({"id": keyword_id, "name": mechanic})

        # Also check oracle text for keywords not in mechanics list
        if "oracle_text" in card and card["oracle_text"]:
            text = card["oracle_text"]

            for keyword in common_keywords:
                # Only process keywords we haven't already found in mechanics list
                # and that appear at the beginning of text, after commas, or after line breaks
                if not any(k["name"] == keyword for k in keywords) and re.search(
                    r"(^|,\s*|\n)" + re.escape(keyword) + r"\b", text
                ):
                    keyword_id = f"keyword_{keyword.lower().replace(' ', '_')}"

                    # Create keyword if it doesn't exist
                    if keyword_id not in self.entities["keywords"]:
                        keyword_obj = {
                            "id": keyword_id,
                            "name": keyword,
                        }
                        self._add_keyword_entity(keyword_obj)

                    keywords.append({"id": keyword_id, "name": keyword})

        return keywords

    def _extract_rule_references(self, text: str) -> List[str]:
        """
        Extract rule references from text.

        Args:
            text: Text to extract rule references from

        Returns:
            List of rule IDs referenced in the text
        """
        referenced_rules = []

        # Look for rule numbers in the format ###.## or ###.##a
        rule_pattern = r"(\d+\.\d+[a-z]?)"
        matches = re.findall(rule_pattern, text)

        for match in matches:
            referenced_rules.append(match)

        return referenced_rules

    def _extract_card_references(self, text: str) -> List[str]:
        """
        Extract card references from text.

        Args:
            text: Text to extract card references from

        Returns:
            List of card IDs referenced in the text
        """
        referenced_cards = []

        # This is a simplified approach - in a real implementation, this would
        # use more sophisticated NLP techniques to identify card names
        for card_name, card_id in self.name_to_id["cards"].items():
            if card_name in text.lower() and card_id not in referenced_cards:
                referenced_cards.append(card_id)

        return referenced_cards

    def _process_glossary(self, glossary_data: Dict[str, Any]) -> None:
        """
        Process glossary data to add terms as keywords or mechanics.

        Args:
            glossary_data: Dictionary of glossary terms
        """
        # Common MTG keywords that should always be treated as keywords, not mechanics
        always_keywords = {
            "flying",
            "first strike",
            "double strike",
            "deathtouch",
            "vigilance",
            "trample",
            "haste",
            "lifelink",
            "flash",
            "reach",
            "defender",
            "hexproof",
            "indestructible",
            "menace",
            "prowess",
            "ward",
            "equip",
        }

        for term, definition in glossary_data.items():
            if not term or not definition:
                continue

            # Determine if this is a keyword or mechanic
            term_type = "keyword"
            term_lower = term.lower()

            # Force common keywords to be keywords
            if term_lower in always_keywords:
                term_type = "keyword"
            # Otherwise use heuristic based on definition
            elif "when" in definition.lower() or "whenever" in definition.lower():
                term_type = "mechanic"

            term_id = f"{term_type}_{term_lower.replace(' ', '_')}"

            if term_type == "keyword":
                # Add as keyword
                if term_id not in self.entities["keywords"]:
                    keyword = {"id": term_id, "name": term, "definition": definition}
                    self._add_keyword_entity(keyword)
            else:
                # Add as mechanic
                if term_id not in self.entities["mechanics"]:
                    mechanic = {"id": term_id, "name": term, "definition": definition}
                    self._add_mechanic_entity(mechanic)

                    # Extract rule references from definition
                    referenced_rules = self._extract_rule_references(definition)
                    for rule_id in referenced_rules:
                        self._add_relationship(
                            "mechanic_governed_by_rule", term_id, rule_id
                        )

    def query(self, query_type: str, **params) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.

        Args:
            query_type: Type of query (e.g., 'entity', 'path', 'neighbors')
            **params: Query parameters

        Returns:
            List of result dictionaries. For path queries, returns the first path found.
        """
        if query_type == "entity":
            return self._query_entity(**params)
        elif query_type == "path":
            # Path query returns list of paths, but our interface expects list of entities
            # So we'll return the first path if available, or an empty list
            paths = self._query_path(**params)
            if paths:
                return paths[0]  # Return the first path found
            return []
        elif query_type == "neighbors":
            return self._query_neighbors(**params)
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return []

    def _query_entity(
        self,
        entity_type: str,
        entity_id: Optional[str] = None,
        entity_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query for entity information.

        Args:
            entity_type: Type of entity (e.g., 'cards', 'rules')
            entity_id: Optional entity ID
            entity_name: Optional entity name

        Returns:
            List of entity dictionaries
        """
        if entity_type not in self.entities:
            logger.warning(f"Unknown entity type: {entity_type}")
            return []

        results = []

        # If ID is provided, look up directly
        if entity_id:
            if entity_id in self.entities[entity_type]:
                results.append(self.entities[entity_type][entity_id])

        # If name is provided, look up by name
        elif entity_name:
            name_lower = entity_name.lower()
            if name_lower in self.name_to_id[entity_type]:
                entity_id = self.name_to_id[entity_type][name_lower]
                results.append(self.entities[entity_type][entity_id])

        # If neither ID nor name is provided, return all entities of this type
        else:
            results = list(self.entities[entity_type].values())

        return results

    def _query_neighbors(
        self,
        entity_type: str,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[Dict[str, Any]]:
        """
        Query for neighboring entities.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            relation_type: Optional relationship type
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of neighboring entity dictionaries
        """
        if (
            entity_type not in self.entities
            or entity_id not in self.entities[entity_type]
        ):
            logger.warning(f"Entity not found: {entity_type}/{entity_id}")
            return []

        neighbors = []

        # Process each relationship type
        rel_types = [relation_type] if relation_type else self.relationships.keys()

        for rel_type in rel_types:
            if rel_type not in self.relationships:
                continue

            for from_id, to_id in self.relationships[rel_type]:
                # Check outgoing relationships
                if (direction in ["outgoing", "both"]) and from_id == entity_id:
                    # Determine target entity type
                    target_type = self._determine_entity_type(to_id)
                    if target_type and to_id in self.entities[target_type]:
                        neighbor = self.entities[target_type][to_id]
                        neighbors.append(
                            {
                                "entity": neighbor,
                                "relation": rel_type,
                                "direction": "outgoing",
                            }
                        )

                # Check incoming relationships
                if (direction in ["incoming", "both"]) and to_id == entity_id:
                    # Determine source entity type
                    source_type = self._determine_entity_type(from_id)
                    if source_type and from_id in self.entities[source_type]:
                        neighbor = self.entities[source_type][from_id]
                        neighbors.append(
                            {
                                "entity": neighbor,
                                "relation": rel_type,
                                "direction": "incoming",
                            }
                        )

        return neighbors

    def _query_path(
        self, from_type: str, from_id: str, to_type: str, to_id: str, max_depth: int = 3
    ) -> List[List[Dict[str, Any]]]:  # Returns a list of paths
        """
        Find paths between two entities.

        Args:
            from_type: Source entity type
            from_id: Source entity ID
            to_type: Target entity type
            to_id: Target entity ID
            max_depth: Maximum path length

        Returns:
            List of paths, where each path is a list of entities and relationships
        """
        if from_type not in self.entities or from_id not in self.entities[from_type]:
            logger.warning(f"Source entity not found: {from_type}/{from_id}")
            return []

        if to_type not in self.entities or to_id not in self.entities[to_type]:
            logger.warning(f"Target entity not found: {to_type}/{to_id}")
            return []

        # Check if source and target are the same
        if from_type == to_type and from_id == to_id:
            entity = self.entities[from_type][from_id]
            return [[{"entity": entity}]]  # Single-node path

        # Use breadth-first search to find paths
        visited_nodes = set()  # Track visited nodes to avoid cycles
        visited_paths = set()  # Track path signatures to avoid duplicate paths

        # Queue holds paths, where each path is a list of nodes
        # Each node is a tuple: (entity_type, entity_id, relation_type, direction, path_signature)
        # Use a more precise typing to avoid type errors
        # Initialize the first path with consistent types
        first_node = (from_type, from_id, None, None, f"{from_type}:{from_id}")
        queue: List[List[Tuple[str, str, Optional[str], Optional[str], str]]] = [
            [first_node]
        ]
        paths = []

        while queue and len(paths) < 10:  # Limit to 10 paths for performance
            path = queue.pop(0)
            current = path[-1]
            current_type, current_id, _, _, path_sig = current

            # Don't revisit the same node in the same path
            node_key = (current_type, current_id)
            if node_key in visited_nodes and len(path) > 1:
                continue

            # Add to visited nodes
            visited_nodes.add(node_key)

            # Check if we've reached the target
            if current_type == to_type and current_id == to_id:
                # Convert path to the expected format
                formatted_path = []
                for i, (e_type, e_id, rel, direction, _) in enumerate(path):
                    entity = self.entities[e_type][e_id]
                    step = {"entity": entity}
                    if i > 0:  # First entity has no incoming relationship
                        step["relation"] = rel
                        step["direction"] = direction
                    formatted_path.append(step)

                # Check if this path is unique before adding
                path_str = str(formatted_path)  # Simple way to check for duplicates
                if path_str not in visited_paths:
                    paths.append(formatted_path)
                    visited_paths.add(path_str)
                continue

            # If max depth reached, skip
            if len(path) > max_depth:
                continue

            # Add neighbors to queue (direct relationships)
            for rel_type, rel_pairs in self.relationships.items():
                # Check both outgoing and incoming relationships
                for from_entity_id, to_entity_id in rel_pairs:
                    # Outgoing relationships
                    if from_entity_id == current_id:
                        # Find entity type for the target
                        target_type = self._determine_entity_type(to_entity_id)
                        if not target_type:
                            continue

                        # Skip if this would create a cycle in the current path
                        if any(
                            step[0] == target_type and step[1] == to_entity_id
                            for step in path
                        ):
                            continue

                        # Create new path with this step
                        new_path_sig = (
                            f"{path_sig}>{rel_type}>{target_type}:{to_entity_id}"
                        )
                        # Cast the new node to ensure consistent typing
                        new_node: Tuple[str, str, Optional[str], Optional[str], str] = (
                            target_type,
                            to_entity_id,
                            rel_type,
                            "outgoing",
                            new_path_sig,
                        )
                        new_path = path + [new_node]
                        queue.append(new_path)

                    # Incoming relationships
                    if to_entity_id == current_id:
                        # Find entity type for the source
                        source_type = self._determine_entity_type(from_entity_id)
                        if not source_type:
                            continue

                        # Skip if this would create a cycle in the current path
                        if any(
                            step[0] == source_type and step[1] == from_entity_id
                            for step in path
                        ):
                            continue

                        # Create new path with this step
                        new_path_sig = (
                            f"{path_sig}>{rel_type}>{source_type}:{from_entity_id}"
                        )
                        # Cast the new node to ensure consistent typing
                        new_node: Tuple[str, str, Optional[str], Optional[str], str] = (
                            source_type,
                            from_entity_id,
                            rel_type,
                            "incoming",
                            new_path_sig,
                        )
                        new_path = path + [new_node]
                        queue.append(new_path)

        return paths

    def _determine_entity_type(self, entity_id: str) -> Optional[str]:
        """
        Determine the entity type from an ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity type or None if not found
        """
        for entity_type in self.entities:
            if entity_id in self.entities[entity_type]:
                return entity_type
        return None

    def get_entity_by_name(
        self, entity_type: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get an entity by its name.

        Args:
            entity_type: Type of entity
            name: Entity name

        Returns:
            Entity dictionary or None if not found
        """
        name_lower = name.lower()
        if (
            entity_type in self.name_to_id
            and name_lower in self.name_to_id[entity_type]
        ):
            entity_id = self.name_to_id[entity_type][name_lower]
            return self.entities[entity_type][entity_id]
        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary of statistics
        """
        return self.stats

    def save_to_file(self, filepath: str) -> None:
        """
        Save the knowledge graph to a file.

        Args:
            filepath: Path to save the file to
        """
        data = {
            "entities": self.entities,
            "relationships": self.relationships,
            "name_to_id": self.name_to_id,
            "schema_version": self.schema_version,
            "last_updated": self.last_updated.isoformat(),
            "stats": self.stats,
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        logger.info(f"Knowledge graph saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """
        Load the knowledge graph from a file.

        Args:
            filepath: Path to load the file from
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.entities = data["entities"]
        self.relationships = data["relationships"]
        self.name_to_id = data["name_to_id"]
        self.schema_version = data["schema_version"]
        self.last_updated = datetime.fromisoformat(data["last_updated"])
        self.stats = data["stats"]

        logger.info(f"Knowledge graph loaded from {filepath}")
        logger.info(
            f"Entities: {sum(self.stats['entity_counts'].values())}, "
            + f"Relationships: {sum(self.stats['relationship_counts'].values())}"
        )
