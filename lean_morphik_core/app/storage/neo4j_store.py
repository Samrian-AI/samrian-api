import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel, Field # Renamed to avoid clash

# Import ABC and generic Node, Relationship from base_graph_store
from .base_graph_store import BaseGraphStore, Node, Relationship

# Assuming neo4j driver is installed
try:
    from neo4j import AsyncGraphDatabase, AsyncSession, AsyncTransaction, Result
    from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
except ImportError:
    AsyncGraphDatabase = None
    AsyncSession = None
    AsyncTransaction = None
    Result = None
    Neo4jError = None
    ServiceUnavailable = None # Handle for tenacity
    SessionExpired = None # Handle for tenacity
    logging.warning("neo4j driver not installed. Neo4jStore will not be functional.")

# Import tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

# Pydantic models for graph elements used internally by Neo4jStore
# Renamed to avoid conflict with generic Node, Relationship from BaseGraphStore
class Neo4jNodeModel(PydanticBaseModel):
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class Neo4jRelationshipModel(PydanticBaseModel):
    source_node_label: str
    source_node_properties: Dict[str, Any]
    target_node_label: str
    target_node_properties: Dict[str, Any]
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class Neo4jStore(BaseGraphStore): # Inherit from BaseGraphStore
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

    # Define retry strategy for Neo4j operations.
    # ServiceUnavailable and SessionExpired are common transient errors.
    # Neo4jError is a base for many other potentially retryable issues.
    _neo4j_retry_conditions = (
        retry_if_exception_type(ServiceUnavailable) |
        retry_if_exception_type(SessionExpired) |
        retry_if_exception_type(Neo4jError) # Be cautious with retrying generic Neo4jError
                                           # if it includes non-transient issues.
                                           # For now, including it for broader coverage.
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_neo4j_retry_conditions,
        before_sleep=before_sleep_log(logger, logging.WARNING), # Log before sleeping on retry
        reraise=True
    )
    async def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        results_list = []
        logger.debug(f"Attempting to execute Neo4j query: {query} with params: {parameters}")
        # The try-except for rollback logic is kept within the retryable block.
        # Tenacity will retry the whole block if Neo4jError (or others specified) occurs.
        # The specific logging of "Neo4j query failed" will happen on the attempt that fails.
        # If all retries fail, tenacity will re-raise the last exception.
        tx: Optional[AsyncTransaction] = None # Ensure tx is defined for finally block
        try:
            async with self._driver.session(database=self.database) as session:
                async with session.begin_transaction() as tx:
                    result_cursor: Result = await tx.run(query, parameters)
                    async for record in result_cursor:
                        results_list.append(record.data())
                    await tx.commit()
                    logger.debug(f"Neo4j query successful: {query}")
            return results_list
        except Neo4jError as e: # Catch Neo4j specific errors for detailed logging before retry
            logger.warning(f"Neo4j query failed on attempt: {query} with params {parameters}. Error: {e}. Will retry if applicable.")
            # Rollback logic for the current failed transaction attempt
            if tx and not tx.closed():
                try:
                    await tx.rollback()
                    logger.info("Transaction rolled back for current attempt.")
                except Exception as rb_exc:
                    logger.error(f"Failed to rollback transaction for current attempt: {rb_exc}")
            raise # Re-raise to allow tenacity to handle retry
        except Exception as e: # Catch other unexpected errors
            logger.error(f"An unexpected error occurred executing Neo4j query: {e}. Query: {query}")
            if tx and not tx.closed():
                 try:
                    await tx.rollback()
                    logger.info("Transaction rolled back due to unexpected error.")
                 except Exception as rb_exc:
                    logger.error(f"Failed to rollback transaction on unexpected error: {rb_exc}")
            raise # Re-raise


    async def add_node(self, node: Node) -> Dict[str, Any]: # Accepts generic Node
        """
        Adds a node to Neo4j. If a node with the same label and properties (key part) exists, it merges.
        Accepts a generic Node object as per BaseGraphStore.
        """
        # This method translates the generic Node into Neo4j specifics.
        # For simplicity, assuming the generic Node's structure is directly usable.
        # If Neo4jNodeModel was used for internal validation:
        # internal_node = Neo4jNodeModel(label=node.label, properties=node.properties)
        # then use internal_node.label, internal_node.properties below.

        if not node.properties:
            cypher_query = f"CREATE (n:`{node.label}`) RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels"
            params = {}
        else:
            merge_props_str = ", ".join([f"`{k}`: $properties.`{k}`" for k in node.properties.keys()])
            cypher_query = (
                f"MERGE (n:`{node.label}` {{{merge_props_str}}}) "
                f"ON CREATE SET n += $properties "
                f"ON MATCH SET n += $properties "
                f"RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels"
            )
            params = {"properties": node.properties}

        logger.info(f"Adding/merging node. Label: {node.label}, Properties: {node.properties}")
        result = await self._execute_query(cypher_query, params)
        return result[0] if result else {}


    async def add_relationship(self, rel: Relationship) -> Dict[str, Any]: # Accepts generic Relationship
        """
        Adds a relationship between two existing nodes. Nodes are matched by label and properties.
        Accepts a generic Relationship object as per BaseGraphStore.
        """
        # Translate generic Relationship to Neo4j specifics.
        # If Neo4jRelationshipModel was used for internal validation:
        # internal_rel = Neo4jRelationshipModel(**rel.dict() if hasattr(rel, 'dict') else vars(rel))
        # then use internal_rel.source_node_label, etc. below.

        source_match_props_str = ", ".join([f"`s`.`{k}` = $source_props.`{k}`" for k in rel.source_node_properties.keys()])
        target_match_props_str = ", ".join([f"`t`.`{k}` = $target_props.`{k}`" for k in rel.target_node_properties.keys()])
        rel_props_str = ", ".join([f"`{k}`: $rel_props.`{k}`" for k in rel.properties.keys()]) if rel.properties else ""

        # Ensure rel_props_str is correctly formatted if empty
        cypher_rel_props = f" {{{rel_props_str}}}" if rel_props_str else ""


        cypher_query = (
            f"MATCH (s:`{rel.source_node_label}`), (t:`{rel.target_node_label}`) "
            f"WHERE {source_match_props_str} AND {target_match_props_str} "
            f"MERGE (s)-[r:`{rel.type}`{cypher_rel_props}]->(t) "
            f"RETURN type(r) as type, properties(r) as properties, "
            f"elementId(startNode(r)) as source_id, elementId(endNode(r)) as target_id"
        )

        params = {
            "source_props": rel.source_node_properties,
            "target_props": rel.target_node_properties,
            "rel_props": rel.properties
        }
        logger.info(f"Adding relationship: {rel.type} from {rel.source_node_label} {rel.source_node_properties} to {rel.target_node_label} {rel.target_node_properties}")
        result = await self._execute_query(cypher_query, params)
        return result[0] if result else {}

    async def get_node(self, label: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]: # Signature matches ABC
        """Retrieves a single node by label and properties."""
        if not properties: # Handle case where properties might be empty
             match_props_str = "true" # Or handle appropriately if Neo4j requires specific syntax for no property match
        else:
            match_props_str = " AND ".join([f"n.`{k}` = ${k}" for k in properties.keys()])

        cypher_query = f"MATCH (n:`{label}`) WHERE {match_props_str} RETURN elementId(n) as id, properties(n) as properties, labels(n) as labels LIMIT 1"

        logger.info(f"Getting node. Label: {label}, Properties: {properties}")
        result = await self._execute_query(cypher_query, properties) # Pass properties directly as params
        return result[0] if result else None

    async def run_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: # Signature matches ABC
        """
        Executes an arbitrary Cypher query and returns the results.
        Use with caution, especially with user-provided queries.
        This method is directly from the ABC.
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
    # For testing, we now use the generic Node and Relationship from the base class
    import os
    # uri = os.environ.get("NEO4J_URI")
    # user = os.environ.get("NEO4J_USER")
    # password = os.environ.get("NEO4J_PASSWORD")

    # if not uri or not user or not password:
    #     print("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.")
    #     return

    # store = Neo4jStore(uri=uri, user=user, password=password)

    # try:
    #     print("Ensuring constraints (example)...")
    #     # await store.ensure_constraints([  # ensure_constraints is not part of BaseGraphStore in this version
    #     #     {"label": "Person", "property": "name"},
    #     #     {"label": "Company", "property": "name"}
    #     # ])

    #     # Add nodes using generic Node
    #     person_node = Node(label="Person", properties={"name": "Alice", "age": 30})
    #     company_node = Node(label="Company", properties={"name": "AcmeCorp", "industry": "Tech"})

    #     alice_result = await store.add_node(person_node)
    #     print(f"Added/Merged Alice: {alice_result}")

    #     acme_result = await store.add_node(company_node)
    #     print(f"Added/Merged AcmeCorp: {acme_result}")

    #     # Add relationship using generic Relationship
    #     works_for_rel = Relationship(
    #         source_node_label="Person",
    #         source_node_properties={"name": "Alice"},
    #         target_node_label="Company",
    #         target_node_properties={"name": "AcmeCorp"},
    #         type="WORKS_FOR",
    #         properties={"role": "Engineer"}
    #     )
    #     rel_result = await store.add_relationship(works_for_rel)
    #     print(f"Added WORKS_FOR relationship: {rel_result}")

    #     # Get a node
    #     retrieved_alice = await store.get_node(label="Person", properties={"name": "Alice"})
    #     print(f"Retrieved Alice: {retrieved_alice}")

    #     # Run custom query
    #     all_persons = await store.run_cypher_query("MATCH (p:Person) RETURN p.name as name, p.age as age")
    #     print(f"All persons: {all_persons}")

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # finally:
    #     await store.close()

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main()) # Comment out for non-testing use
    pass
