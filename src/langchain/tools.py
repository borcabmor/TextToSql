import datetime as dt
import uuid

from sqlalchemy import create_engine, inspect, text

from langchain.tools import tool
from src.retrieve import SQLRetriever


def make_retrieve_sql(retriever: SQLRetriever):
    @tool
    def retrieve_sql(question: str) -> str:
        """
        Retrieve a semantically similar SQL example to guide SQL generation.
        """

        results = retriever.retrieve(question)

        return results[0]["sql"] if results else ""

    return retrieve_sql


@tool
def get_db_schema(connection_string: str) -> str:
    """
    Inspect a PostgreSQL database and return schema information.

    Includes:
    - tables
    - columns
    - primary keys
    - foreign keys
    """

    if "sslmode=" not in connection_string:
        connection_string += (
            "&sslmode=require" if "?" in connection_string else "?sslmode=require"
        )

    engine = create_engine(connection_string)

    try:
        inspector = inspect(engine)
        schema_parts = []

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)

            col_defs = []

            for col in columns:
                col_def = f"{col['name']} {col['type']}"

                if col["name"] in pk.get("constrained_columns", []):
                    col_def += " PK"

                col_defs.append(col_def)

            for fk in fks:
                col_defs.append(
                    f"FK({', '.join(fk['constrained_columns'])}) "
                    f"-> {fk['referred_table']}({', '.join(fk['referred_columns'])})"
                )

            schema_parts.append(f"{table_name}({', '.join(col_defs)})")

        return "\n".join(schema_parts)

    finally:
        engine.dispose()


def make_execute_sql(connection_string: str):
    @tool
    def execute_sql(sql: str):
        """
        Execute a SQL query and return results as a list of dictionaries.
        """

        cs = connection_string

        if "sslmode=" not in cs:
            cs += "&sslmode=require" if "?" in cs else "?sslmode=require"

        engine = create_engine(cs)

        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(r._mapping) for r in result]

                def serialize(v):
                    if isinstance(v, (dt.datetime, dt.date)):
                        return v.isoformat()
                    if isinstance(v, uuid.UUID):
                        return str(v)

                    return v

                return [{k: serialize(v) for k, v in row.items()} for row in rows]

        finally:
            engine.dispose()

    return execute_sql
