# src/rag/graph.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, lit
from graphframes import GraphFrame
from typing import List
from langchain_core.documents import Document

from utils.config import config_manager as databricks_config
from .llm_provider import get_llm

def get_spark_session() -> SparkSession:
    """Get or create a Spark session configured for the project."""
    return SparkSession.builder \
        .appName("Maverick GraphRAG") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def extract_entities_and_relationships(text_chunk: str) -> dict:
    """
    Uses an LLM to extract entities (nodes) and relationships (edges) from a text chunk.
    This is a simplified example. A production system would use a more robust prompt
    and potentially a model fine-tuned for information extraction.
    """
    llm = get_llm()
    prompt = f"""
    From the following text, extract named entities and their relationships.
    Provide the output as a JSON object with two keys: "entities" and "relationships".
    - "entities" should be a list of strings.
    - "relationships" should be a list of triplets, where each triplet is [source_entity, relationship_type, target_entity].

    Example:
    Text: "The project uses Databricks for compute and Unity Catalog for governance."
    Output: {{"entities": ["Databricks", "Unity Catalog"], "relationships": [["Databricks", "is_used_for", "compute"], ["Unity Catalog", "is_used_for", "governance"]]}}

    Now, process this text:
    "{text_chunk}"
    """
    try:
        response = llm.invoke(prompt)
        # A real implementation would have more robust JSON parsing and error handling
        import json
        return json.loads(response.content)
    except Exception:
        return {"entities": [], "relationships": []}

def build_and_save_graph(splits: List[Document]):
    """
    Builds a knowledge graph from document splits using Spark GraphFrames
    and saves the vertices and edges to Delta tables.
    """
    print("🧠 Building Knowledge Graph with Spark GraphFrames...")
    spark = get_spark_session()

    all_entities = set()
    all_relationships = []

    for doc in splits:
        graph_data = extract_entities_and_relationships(doc.page_content)
        for entity in graph_data.get("entities", []):
            all_entities.add((entity,))
        for rel in graph_data.get("relationships", []):
            # Ensure relationships are triplets
            if isinstance(rel, list) and len(rel) == 3:
                all_relationships.append(tuple(rel))

    if not all_entities or not all_relationships:
        print("⚠️ No entities or relationships extracted. Skipping graph creation.")
        return

    # Create Vertices DataFrame
    vertices = spark.createDataFrame(list(all_entities), ["id"])
    vertices_table = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.graph_vertices"
    vertices.write.format("delta").mode("overwrite").saveAsTable(vertices_table)
    print(f"✅ Saved {vertices.count()} vertices to {vertices_table}")

    # Create Edges DataFrame
    edges = spark.createDataFrame(all_relationships, ["src", "relationship", "dst"])
    edges_table = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.graph_edges"
    edges.write.format("delta").mode("overwrite").saveAsTable(edges_table)
    print(f"✅ Saved {edges.count()} edges to {edges_table}")

    # (Optional) Create and display the GraphFrame object
    # graph = GraphFrame(vertices, edges)
    # print("GraphFrame Summary:")
    # graph.vertices.show(5)
    # graph.edges.show(5)

class GraphRAGRetriever:
    """A retriever that queries the GraphFrame-based knowledge graph."""
    def __init__(self):
        self.spark = get_spark_session()
        self.vertices_table = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.graph_vertices"
        self.edges_table = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.graph_edges"
        
        try:
            vertices = self.spark.read.table(self.vertices_table)
            edges = self.spark.read.table(self.edges_table)
            self.graph = GraphFrame(vertices, edges)
            print("✅ GraphRAG Retriever initialized successfully.")
        except Exception as e:
            print(f"❌ Could not initialize GraphRAG Retriever. Have you run the ingestion pipeline? Error: {e}")
            self.graph = None

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        if not self.graph:
            return []

        # Simple keyword matching on entities (a real implementation would be more complex)
        keywords = query.lower().split()
        
        # Find entities that match keywords in the query
        matching_entities = self.graph.vertices.filter(
            col("id").rlike(f"(?i)({'|'.join(keywords)})")
        ).limit(k)

        if matching_entities.count() == 0:
            return []

        # Find paths of length 1 connected to the matching entities
        # This finds direct relationships (e.g., entity -> relationship -> other_entity)
        paths = self.graph.find("(a)-[e]->(b)").filter(
            col("a.id").isin([row.id for row in matching_entities.collect()])
        )
        
        context_str = "\n".join(
            [f"- {row.a.id} {row.e.relationship} {row.b.id}" for row in paths.collect()]
        )

        if not context_str:
            return [Document(page_content="No direct relationships found for the query entities.")]

        return [Document(page_content=f"Found the following relationships in the knowledge graph:\n{context_str}")]