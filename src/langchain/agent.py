import json
import logging
from typing import Any, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents import create_agent
from src.langchain.prompts import SYSTEM_PROMPT
from src.langchain.tools import get_db_schema, make_execute_sql, make_retrieve_sql
from src.retrieve import SQLRetriever


class AgentState(TypedDict):
    question: str
    connection_string: str
    schema: Optional[str]
    sql: Optional[str]
    result: Optional[Any]


class TextToSQLAgent:
    def __init__(
        self,
        retriever: SQLRetriever,
        llm_model: str = "claude-haiku-4-5",
    ):
        self.logger = logging.getLogger(__name__)
        self.model = ChatAnthropic(model=llm_model)
        self.retrieve_tool = make_retrieve_sql(retriever)
        self.schema_cache = {}

    def load_schema_node(self, state: AgentState) -> str:
        """
        Load schema once per request and store in state
        """

        if state.get("schema"):
            return state["schema"]

        cs = state["connection_string"]

        if cs in self.schema_cache:
            self.logger.info("Schema loaded from cache")
            state["schema"] = self.schema_cache[cs]

            return state["schema"]

        self.logger.info("Loading schema from DB")

        schema = get_db_schema.invoke({"connection_string": cs})

        self.schema_cache[cs] = schema
        state["schema"] = schema

        return schema

    def generate_sql(self, state: AgentState) -> str:
        """
        Generate SQL only
        """

        agent = create_agent(
            model=self.model,
            tools=[self.retrieve_tool],
            system_prompt=SYSTEM_PROMPT,
        )

        response = agent.invoke({"messages": [HumanMessage(content=f"""
                Schema:
                {state['schema']}

                Question:
                {state['question']}

                Return ONLY SQL.
                """)]})

        sql = response["messages"][-1].content.strip()

        # Remove markdown chars
        if sql.startswith("```"):
            sql = sql.removeprefix("```sql")
            sql = sql.removeprefix("```")
            sql = sql.removesuffix("```")
            sql = sql.strip()

        state["sql"] = sql

        return state["sql"]

    def query(self, state: AgentState) -> AgentState:
        """
        Execute SQL query
        """

        exec_agent = create_agent(
            model=self.model,
            tools=[
                self.retrieve_tool,
                make_execute_sql(state["connection_string"]),
            ],
            system_prompt=SYSTEM_PROMPT,
        )

        response = exec_agent.invoke(
            {"messages": [HumanMessage(content=f"""
                Schema:
                {state['schema']}

                Question:
                {state['question']}

                Generate SQL and call execute_sql.
                Return no explanation.
                """)]},
            config={"recursion_limit": 5},
        )

        self.logger.info(response)

        # SQL usado
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if tc["name"] == "execute_sql":
                        state["sql"] = tc["args"].get("sql")

        # Resultado
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage) and msg.name == "execute_sql":
                try:
                    state["result"] = json.loads(msg.content)
                except Exception:
                    state["result"] = msg.content

                break

        return state
