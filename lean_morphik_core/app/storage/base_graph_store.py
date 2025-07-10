from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Placeholder for Node and Relationship structures, similar to DocumentVector
# These would ideally be defined in a common models area if they are complex
# or defined simply here if their structure is straightforward for the interface.

class Node: # Generic placeholder
    label: str
    properties: Dict[str, Any]

    def __init__(self, label: str, properties: Dict[str, Any]):
        self.label = label
        self.properties = properties

class Relationship: # Generic placeholder
    source_node_label: str
    source_node_properties: Dict[str, Any]
    target_node_label: str
    target_node_properties: Dict[str, Any]
    type: str
    properties: Dict[str, Any]

    def __init__(self, source_node_label: str, source_node_properties: Dict[str, Any],
                 target_node_label: str, target_node_properties: Dict[str, Any],
                 type: str, properties: Dict[str, Any]):
        self.source_node_label = source_node_label
        self.source_node_properties = source_node_properties
        self.target_node_label = target_node_label
        self.target_node_properties = target_node_properties
        self.type = type
        self.properties = properties


class BaseGraphStore(ABC):
    """
    Abstract base class for graph store operations.
    """

    @abstractmethod
    async def add_node(self, node: Node) -> Dict[str, Any]:
        """
        Adds a node to the graph store.

        Args:
            node: A Node object representing the node to add.

        Returns:
            A dictionary representing the created/merged node from the graph store.
        """
        pass

    @abstractmethod
    async def add_relationship(self, relationship: Relationship) -> Dict[str, Any]:
        """
        Adds a relationship between two nodes in the graph store.

        Args:
            relationship: A Relationship object representing the relationship to add.

        Returns:
            A dictionary representing the created/merged relationship from the graph store.
        """
        pass

    @abstractmethod
    async def get_node(self, label: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single node by label and properties.

        Args:
            label: The label of the node.
            properties: Key-value properties to match the node.

        Returns:
            A dictionary representing the found node, or None if not found.
        """
        pass

    @abstractmethod
    async def run_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes an arbitrary Cypher query (or graph query language specific to the store)
        and returns the results.

        Args:
            query: The query string.
            parameters: Optional parameters for the query.

        Returns:
            A list of dictionaries, where each dictionary is a result row.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Closes any open connections to the graph store.
        """
        pass

    # Optional: Add other common graph store operations
    # @abstractmethod
    # async def ensure_constraints(self, constraints: List[Dict[str, str]]):
    #     pass
