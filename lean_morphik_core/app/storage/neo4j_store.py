import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Assuming neo4j driver is installed
try:
    from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction, Result
    from neo4j.exceptions import Neo4jError
except ImportError:
    AsyncGraphDatabase = None
    AsyncSession = None
    AsyncTransaction = None
    Result = None
    Neo4jError = None
    logging.warning("neo4j driver not installed. Neo4jStore will not be functional.")

logger = logging.getLogger(__name__)

# Pydantic models for graph elements
class Neo4jNode(BaseModel):
    label: str # Primary label for the node
    properties: Dict[str, Any] = Field(default_factory=dict)
    # You might add an optional id_property_key if nodes are uniquely identified by a property other than Neo4j's internal ID.
    # For example, id_property_key: str = "name"

class Neo4jRelationship(BaseModel):
    source_node_label: str
    source_node_properties: Dict[str, Any] # Properties to match the source node
    target_node_label: str
    target_node_properties: Dict[str, Any] # Properties to match the target node
    type: str # Relationship type
    properties: Dict[str, Any] = Field(default_factory=dict)


class Neo4jStore:
    def __init__(self, uri: str, user: Optional[str] = None, password: Optional[str] = None, database: str = "neo4j"):
        if AsyncGraphDatabase is None:
            raise ImportError("Neo4j driver is not installed. Please install with `pip install neo4j`")

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password) if user and password else None)
        self.database = database
        logger.info(f"Neo4jStore initialized for URI: {uri}, Database: {database}")

    async def close(self):
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j driver closed.")

    async def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        results_list = []
        try:
            async with self._driver.session(database=self.database) as session:
                tx: AsyncTransaction
                async with session.begin_transaction() as tx:
                    result_cursor: Result = await tx.run(query, parameters)
                    async for record in result_cursor:
                        results_list.append(record.data())
                    await tx.commit()
            return results_list
        except Neo4jError as e:
            logger.error(f"Neo4j query failed: {query} with params {parameters}. Error: {e}")
            if 'tx' in locals() and tx.closed() is False: # Check if tx was defined and is not closed
                try:
                    await tx.rollback()
                    logger.info("Transaction rolled back.")
                except Exception as rb_exc:
                    logger.error(f"Failed to rollback transaction: {rb_exc}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred executing Neo4j query: {e}")
            if 'tx' in locals() and tx.closed() is False:
                 try:
                    await tx.rollback()
                    logger.info("Transaction rolled back.")
                 except Exception as rb_exc:
                    logger.error(f"Failed to rollback transaction: {rb_exc}")
            raise


    async def add_node(self, node: Neo4jNode) -> Dict[str, Any]:
        """
        Adds a node to Neo4j. If a node with the same label and properties (key part) exists, it merges.
        Assumes properties in Neo4jNode.properties are the ones to match for MERGE.
        A more robust way would be to define a unique key property.
        """
        # Basic MERGE based on all properties. For specific unique keys, adjust the query.
        # Example: MERGE (n:Label {unique_id: $props.unique_id}) ON CREATE SET n = $props RETURN n

        # Constructing the SET part of the Cypher query
        # Ensure that property keys are valid Cypher identifiers (e.g., no spaces, special chars unless quoted)
        # For simplicity, this example assumes valid property keys.
        prop_match_string = ", ".join([f"`{k}`: ${k}" for k in node.properties.keys()])

        # Query to merge based on label and all provided properties
        # This means if any property differs, a new node will be created unless properties for matching are specified.
        # A common pattern is to MERGE on a unique ID property.
        # For this example, we MERGE on the combination of all properties.

        # Let's assume the first property is a unique identifier for MERGE for simplicity
        # This is a placeholder; a proper unique ID strategy is needed for robust merging.
        if not node.properties:
            query = f"CREATE (n:`{node.label}`) RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels"
            params = {}
        else:
            # A simple MERGE based on all properties.
            # For production, you'd typically MERGE on a unique key (e.g., MERGE (n:Label {id_key: $id_val}))
            merge_props_str = ", ".join([f"`{k}`: $properties.`{k}`" for k in node.properties.keys()])
            query = (
                f"MERGE (n:`{node.label}` {{{merge_props_str}}}) "
                f"ON CREATE SET n += $properties " # Use += to ensure existing properties are not overwritten if merging by a subset
                f"ON MATCH SET n += $properties "  # Update properties on match as well
                f"RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels"
            )
            params = {"properties": node.properties}

        logger.info(f"Adding/merging node. Label: {node.label}, Properties: {node.properties}")
        result = await self._execute_query(query, params)
        return result[0] if result else {}


    async def add_relationship(self, rel: Neo4jRelationship) -> Dict[str, Any]:
        """
        Adds a relationship between two existing nodes. Nodes are matched by label and properties.
        """
        source_match_props_str = ", ".join([f"`s`.`{k}` = $source_props.`{k}`" for k in rel.source_node_properties.keys()])
        target_match_props_str = ", ".join([f"`t`.`{k}` = $target_props.`{k}`" for k in rel.target_node_properties.keys()])

        rel_props_str = ", ".join([f"`{k}`: $rel_props.`{k}`" for k in rel.properties.keys()])

        query = (
            f"MATCH (s:`{rel.source_node_label}`), (t:`{rel.target_node_label}`) "
            f"WHERE {source_match_props_str} AND {target_match_props_str} "
            f"MERGE (s)-[r:`{rel.type}` {{{rel_props_str}}}]->(t) "
            f"RETURN type(r) as type, properties(r) as properties, "
            f"elementId(startNode(r)) as source_id, elementId(endNode(r)) as target_id"
        )

        params = {
            "source_props": rel.source_node_properties,
            "target_props": rel.target_node_properties,
            "rel_props": rel.properties
        }
        logger.info(f"Adding relationship: {rel.type} from {rel.source_node_label} {rel.source_node_properties} to {rel.target_node_label} {rel.target_node_properties}")
        result = await self._execute_query(query, params)
        return result[0] if result else {}

    async def get_node(self, label: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieves a single node by label and properties."""
        match_props_str = " AND ".join([f"n.`{k}` = ${k}" for k in properties.keys()])
        query = f"MATCH (n:`{label}`) WHERE {match_props_str} RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels LIMIT 1"

        logger.info(f"Getting node. Label: {label}, Properties: {properties}")
        result = await self._execute_query(query, properties)
        return result[0] if result else None

    async def run_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes an arbitrary Cypher query and returns the results.
        Use with caution, especially with user-provided queries.
        """
        logger.info(f"Running custom Cypher query: {query} with params: {parameters}")
        return await self._execute_query(query, parameters)

    async def ensure_constraints(self, constraints: List[Dict[str, str]]):
        """
        Ensures unique constraints are set up.
        Each dict in constraints should be: {"label": "NodeLabel", "property": "propertyName"}
        """
        # This is a simplified version. Production systems might need more robust constraint management.
        # Note: Neo4j requires Enterprise Edition for node property existence constraints on multiple properties or composite unique constraints.
        # This example creates unique constraints on single properties.
        async with self._driver.session(database=self.database) as session:
            for constraint in constraints:
                label = constraint["label"]
                prop = constraint["property"]
                query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.`{prop}` IS UNIQUE"
                try:
                    await session.run(query)
                    logger.info(f"Ensured unique constraint on :{label}({prop})")
                except Neo4jError as e:
                    # It's okay if the constraint already exists, but log other errors.
                    if "already exists" not in str(e).lower(): # Crude check, specific error codes are better
                        logger.error(f"Failed to create constraint for :{label}({prop}): {e}")
                        raise # Re-raise if it's not an "already exists" type of error

# Example Usage (for testing purposes)
async def main():
    # This requires a running Neo4j instance and NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars
    import os
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri or not user or not password:
        print("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
        return

    store = Neo4jStore(uri=uri, user=user, password=password)

    try:
        print("Ensuring constraints (example)...")
        await store.ensure_constraints([
            {"label": "Person", "property": "name"},
            {"label": "Company", "property": "name"}
        ])

        # Add nodes
        person_node = Neo4jNode(label="Person", properties={"name": "Alice", "age": 30})
        company_node = Neo4jNode(label="Company", properties={"name": "AcmeCorp", "industry": "Tech"})

        alice_result = await store.add_node(person_node)
        print(f"Added/Merged Alice: {alice_result}")

        acme_result = await store.add_node(company_node)
        print(f"Added/Merged AcmeCorp: {acme_result}")

        # Add relationship
        works_for_rel = Neo4jRelationship(
            source_node_label="Person",
            source_node_properties={"name": "Alice"},
            target_node_label="Company",
            target_node_properties={"name": "AcmeCorp"},
            type="WORKS_FOR",
            properties={"role": "Engineer"}
        )
        rel_result = await store.add_relationship(works_for_rel)
        print(f"Added WORKS_FOR relationship: {rel_result}")

        # Get a node
        retrieved_alice = await store.get_node(label="Person", properties={"name": "Alice"})
        print(f"Retrieved Alice: {retrieved_alice}")

        # Run custom query
        all_persons = await store.run_cypher_query("MATCH (p:Person) RETURN p.name as name, p.age as age")
        print(f"All persons: {all_persons}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await store.close()

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main()) # Comment out for non-testing use
    pass
