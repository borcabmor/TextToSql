from langchain.tools import tool
from sqlalchemy import create_engine, inspect, text
from src.retrieve import SQLRetriever


def make_retrieve_sql(retriever: SQLRetriever):
    @tool
    def retrieve_sql(question: str) -> str:
        """Retrieve a similar SQL example."""
        results = retriever.retrieve(question)

        return results[0]["sql"] if results else ""

    return retrieve_sql


@tool
def get_db_schema(connection_string: str) -> str:
    """Return database schema."""

    engine = create_engine(connection_string)
    inspector = inspect(engine)

    schema_parts = []

    for table_name in inspector.get_table_names():

        columns = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)
        fks = inspector.get_foreign_keys(table_name)

        col_defs = []

        for col in columns:
            col_def = f"{col['name']} {col['type']}"

            if col["name"] in pk.get(
                "constrained_columns",
                [],
            ):
                col_def += " PRIMARY KEY"

            col_defs.append(col_def)

        for fk in fks:
            col_defs.append(
                f"FOREIGN KEY ({', '.join(fk['constrained_columns'])}) "
                f"REFERENCES {fk['referred_table']}({', '.join(fk['referred_columns'])})"
            )

        schema_parts.append(
            f"CREATE TABLE {table_name} (\n" f"{', '.join(col_defs)}\n);"
        )

    return "\n\n".join(schema_parts)


@tool
def execute_sql(
    connection_string: str,
    sql: str,
) -> list[dict]:
    """Execute SQL query."""

    engine = create_engine(connection_string)

    with engine.connect() as conn:
        result = conn.execute(text(sql))

        return [dict(row._mapping) for row in result]
