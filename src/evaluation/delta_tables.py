"""
Delta table management for telemetry and evaluation data in Unity Catalog.
"""
import os
from typing import Dict, Any, List, Optional
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, 
    TimestampType, BooleanType, MapType, ArrayType
)
from pyspark.sql.functions import col, lit, current_timestamp, from_json
from delta.tables import DeltaTable
import json

from src.utils.config import config_manager as databricks_config


class DeltaTableManager:
    """Manages Delta tables for telemetry and evaluation data."""
    
    def __init__(self):
        self.spark = None
        self.catalog_name = databricks_config.databricks.catalog
        self.schema_name = databricks_config.databricks.schema
    
    def get_spark_session(self) -> SparkSession:
        """Get or create Spark session."""
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("Maverick RAG Telemetry") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
        return self.spark
    
    def create_telemetry_tables(self):
        """Create all telemetry-related Delta tables."""
        spark = self.get_spark_session()
        
        # Create user interactions table
        self._create_user_interactions_table(spark)
        
        # Create sessions table
        self._create_sessions_table(spark)
        
        # Create conversations table
        self._create_conversations_table(spark)
        
        # Create turns table
        self._create_turns_table(spark)
        
        # Create evaluation results table
        self._create_evaluation_results_table(spark)
        
        # Create performance metrics table
        self._create_performance_metrics_table(spark)
        
        print("✅ All telemetry Delta tables created successfully")
    
    def _create_user_interactions_table(self, spark: SparkSession):
        """Create user interactions Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.user_interactions"
        
        schema = StructType([
            StructField("event_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("conversation_id", StringType(), False),
            StructField("turn_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("interaction_type", StringType(), False),
            StructField("event_type", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("data", MapType(StringType(), StringType()), True),
            StructField("metadata", MapType(StringType(), StringType()), True),
            StructField("created_at", TimestampType(), False)
        ])
        
        # Create empty DataFrame with schema
        empty_df = spark.createDataFrame([], schema)
        
        # Write to Delta table
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def _create_sessions_table(self, spark: SparkSession):
        """Create sessions Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.sessions"
        
        schema = StructType([
            StructField("session_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), True),
            StructField("duration_seconds", DoubleType(), True),
            StructField("interaction_count", IntegerType(), False),
            StructField("last_activity", TimestampType(), True),
            StructField("is_active", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("updated_at", TimestampType(), False)
        ])
        
        empty_df = spark.createDataFrame([], schema)
        
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def _create_conversations_table(self, spark: SparkSession):
        """Create conversations Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.conversations"
        
        schema = StructType([
            StructField("conversation_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), True),
            StructField("duration_seconds", DoubleType(), True),
            StructField("turn_count", IntegerType(), False),
            StructField("last_activity", TimestampType(), True),
            StructField("is_active", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("updated_at", TimestampType(), False)
        ])
        
        empty_df = spark.createDataFrame([], schema)
        
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def _create_turns_table(self, spark: SparkSession):
        """Create turns Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.turns"
        
        schema = StructType([
            StructField("turn_id", StringType(), False),
            StructField("conversation_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), True),
            StructField("duration_seconds", DoubleType(), True),
            StructField("query", StringType(), True),
            StructField("response", StringType(), True),
            StructField("response_time_seconds", DoubleType(), True),
            StructField("query_length", IntegerType(), True),
            StructField("response_length", IntegerType(), True),
            StructField("is_complete", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("updated_at", TimestampType(), False)
        ])
        
        empty_df = spark.createDataFrame([], schema)
        
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def _create_evaluation_results_table(self, spark: SparkSession):
        """Create evaluation results Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.evaluation_results"
        
        schema = StructType([
            StructField("evaluation_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("conversation_id", StringType(), False),
            StructField("turn_id", StringType(), False),
            StructField("query", StringType(), False),
            StructField("expected_answer", StringType(), False),
            StructField("actual_answer", StringType(), False),
            StructField("precision", DoubleType(), False),
            StructField("recall", DoubleType(), False),
            StructField("accuracy", DoubleType(), False),
            StructField("f1_score", DoubleType(), False),
            StructField("semantic_similarity", DoubleType(), False),
            StructField("answer_quality_score", DoubleType(), False),
            StructField("response_time_seconds", DoubleType(), False),
            StructField("num_retrieved_docs", IntegerType(), False),
            StructField("num_relevant_docs", IntegerType(), False),
            StructField("avg_relevance_score", DoubleType(), False),
            StructField("evaluation_timestamp", TimestampType(), False),
            StructField("created_at", TimestampType(), False)
        ])
        
        empty_df = spark.createDataFrame([], schema)
        
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def _create_performance_metrics_table(self, spark: SparkSession):
        """Create performance metrics Delta table."""
        table_name = f"{self.catalog_name}.{self.schema_name}.performance_metrics"
        
        schema = StructType([
            StructField("metric_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("conversation_id", StringType(), False),
            StructField("turn_id", StringType(), False),
            StructField("metric_name", StringType(), False),
            StructField("metric_value", DoubleType(), False),
            StructField("metric_unit", StringType(), True),
            StructField("metric_category", StringType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True),
            StructField("timestamp", TimestampType(), False),
            StructField("created_at", TimestampType(), False)
        ])
        
        empty_df = spark.createDataFrame([], schema)
        
        empty_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_name)
        
        print(f"✅ Created table: {table_name}")
    
    def insert_user_interaction(self, interaction_data: Dict[str, Any]):
        """Insert a user interaction record."""
        spark = self.get_spark_session()
        table_name = f"{self.catalog_name}.{self.schema_name}.user_interactions"
        
        # Prepare data
        data = {
            "event_id": interaction_data["event_id"],
            "session_id": interaction_data["session_id"],
            "conversation_id": interaction_data["conversation_id"],
            "turn_id": interaction_data["turn_id"],
            "user_id": interaction_data.get("user_id"),
            "interaction_type": interaction_data["interaction_type"],
            "event_type": interaction_data["event_type"],
            "timestamp": interaction_data["timestamp"],
            "data": json.dumps(interaction_data.get("data", {})),
            "metadata": json.dumps(interaction_data.get("metadata", {})),
            "created_at": current_timestamp()
        }
        
        # Create DataFrame and insert
        df = spark.createDataFrame([data])
        df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(table_name)
    
    def insert_evaluation_result(self, evaluation_data: Dict[str, Any]):
        """Insert an evaluation result record."""
        spark = self.get_spark_session()
        table_name = f"{self.catalog_name}.{self.schema_name}.evaluation_results"
        
        # Prepare data
        data = {
            "evaluation_id": evaluation_data.get("evaluation_id", "default"),
            "session_id": evaluation_data.get("session_id", "default"),
            "conversation_id": evaluation_data.get("conversation_id", "default"),
            "turn_id": evaluation_data.get("turn_id", "default"),
            "query": evaluation_data["query"],
            "expected_answer": evaluation_data["expected_answer"],
            "actual_answer": evaluation_data["actual_answer"],
            "precision": evaluation_data["precision"],
            "recall": evaluation_data["recall"],
            "accuracy": evaluation_data["accuracy"],
            "f1_score": evaluation_data["f1_score"],
            "semantic_similarity": evaluation_data["semantic_similarity"],
            "answer_quality_score": evaluation_data["answer_quality_score"],
            "response_time_seconds": evaluation_data["response_time_seconds"],
            "num_retrieved_docs": evaluation_data.get("num_retrieved_docs", 0),
            "num_relevant_docs": evaluation_data.get("num_relevant_docs", 0),
            "avg_relevance_score": evaluation_data.get("avg_relevance_score", 0.0),
            "evaluation_timestamp": current_timestamp(),
            "created_at": current_timestamp()
        }
        
        # Create DataFrame and insert
        df = spark.createDataFrame([data])
        df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(table_name)
    
    def insert_performance_metric(self, metric_data: Dict[str, Any]):
        """Insert a performance metric record."""
        spark = self.get_spark_session()
        table_name = f"{self.catalog_name}.{self.schema_name}.performance_metrics"
        
        # Prepare data
        data = {
            "metric_id": metric_data.get("metric_id", "default"),
            "session_id": metric_data.get("session_id", "default"),
            "conversation_id": metric_data.get("conversation_id", "default"),
            "turn_id": metric_data.get("turn_id", "default"),
            "metric_name": metric_data["metric_name"],
            "metric_value": metric_data["metric_value"],
            "metric_unit": metric_data.get("metric_unit"),
            "metric_category": metric_data.get("metric_category"),
            "metadata": json.dumps(metric_data.get("metadata", {})),
            "timestamp": current_timestamp(),
            "created_at": current_timestamp()
        }
        
        # Create DataFrame and insert
        df = spark.createDataFrame([data])
        df.write \
            .format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(table_name)
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a Delta table."""
        spark = self.get_spark_session()
        full_table_name = f"{self.catalog_name}.{self.schema_name}.{table_name}"
        
        try:
            # Get table schema
            df = spark.table(full_table_name)
            schema = df.schema
            
            # Get row count
            row_count = df.count()
            
            # Get table properties
            table_properties = spark.sql(f"DESCRIBE DETAIL {full_table_name}").collect()
            
            return {
                "table_name": full_table_name,
                "schema": [{"name": field.name, "type": str(field.dataType)} for field in schema.fields],
                "row_count": row_count,
                "properties": [row.asDict() for row in table_properties]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old telemetry data."""
        spark = self.get_spark_session()
        
        tables = [
            "user_interactions",
            "sessions", 
            "conversations",
            "turns",
            "evaluation_results",
            "performance_metrics"
        ]
        
        for table_name in tables:
            full_table_name = f"{self.catalog_name}.{self.schema_name}.{table_name}"
            try:
                # Delete records older than specified days
                spark.sql(f"""
                    DELETE FROM {full_table_name} 
                    WHERE created_at < current_timestamp() - INTERVAL {days_to_keep} DAYS
                """)
                print(f"✅ Cleaned up old data from {table_name}")
            except Exception as e:
                print(f"❌ Error cleaning up {table_name}: {e}")
    
    def close(self):
        """Close Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None


# Global Delta table manager instance
delta_table_manager = DeltaTableManager()
