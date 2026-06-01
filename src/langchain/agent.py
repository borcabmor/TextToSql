import logging

from langchain_anthropic import ChatAnthropic

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from src.langchain.prompts import SYSTEM_PROMPT
from src.langchain.tools import execute_sql, get_db_schema, make_retrieve_sql
from src.retrieve import SQLRetriever


class TextToSQLAgent:
    def __init__(self, retriever: SQLRetriever, llm_model: str = "claude-sonnet-4-6"):
        self.logger = logging.getLogger(__name__)

        model = ChatAnthropic(model=llm_model)

        self.agent = create_agent(
            model=model,
            tools=[make_retrieve_sql(retriever), get_db_schema, execute_sql],
            system_prompt=SYSTEM_PROMPT,
        )

    def generate_sql(self, question: str, connection_string: str) -> str:
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=f"""
                Connection string:
                {connection_string}
                Question:
                {question}
                Generate SQL only.
                """)]},
            debug=True,
        )

        self.logger.info(response)

        return response["messages"][-1].content.strip()

    def query(self, question: str, connection_string: str):
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=f"""
            Connection string:
            {connection_string}
            Question:
            {question}
            Generate and execute SQL.
            """)]},
            debug=True,
        )

        self.logger.info(response)

        return {"response": response["messages"][-1].content}
