import streamlit as st
import os
import json
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
import pandas as pd
from datetime import datetime, timezone

# --- 1. Configuration and Initialization ---

# Load credentials from environment variables
try:
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    
    COSMOS_DB_ENDPOINT = os.environ["COSMOS_DB_ENDPOINT"]
    COSMOS_DB_KEY = os.environ["COSMOS_DB_KEY"]
    COSMOS_DB_DATABASE_NAME = os.environ["COSMOS_DB_DATABASE_NAME"]
    COSMOS_DB_COLLECTION_NAME = os.environ["COSMOS_DB_COLLECTION_NAME"]
except KeyError as e:
    st.error(f"FATAL ERROR: Environment variable {e} is not set.")
    st.stop()

# Initialize clients
@st.cache_resource
def get_clients():
    oai_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, credential=COSMOS_DB_KEY)
    database = cosmos_client.get_database_client(COSMOS_DB_DATABASE_NAME)
    container = database.get_container_client(COSMOS_DB_COLLECTION_NAME)
    return oai_client, container

oai_client, container = get_clients()

# --- 2. Core "Text-to-SQL" Agent Logic ---

def generate_cosmos_sql(user_question: str, container_schema: dict) -> str:
    """Uses an LLM to generate a Cosmos DB SQL query from a natural language question."""
    
    # Get today's date for context if the user asks for "today"
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    system_prompt = f"""
    You are an expert AI data analyst that translates natural language questions into valid Azure Cosmos DB for NoSQL queries.
    Your goal is to generate a single, executable SQL query string and nothing else.

    DATABASE SCHEMA:
    - Container Name: c
    - Available Columns: {json.dumps(container_schema, indent=2)}

    QUERY GENERATION RULES:
    1.  **Vector Search**: If the question is semantic or conceptual (e.g., "tell me about", "what are some events related to"), use the `VectorDistance` function on the `c.contentVector` field. The query should look like: `SELECT TOP 5 c.id, c.content, c.source_url FROM c ORDER BY VectorDistance(c.contentVector, @query_vector)`
    2.  **Aggregate/Fact-based Queries**: If the question asks for a specific number, count, max, min, or average, generate a standard SQL query.
        - For counts, use `SELECT VALUE COUNT(1) FROM c ...`
        - For other aggregations, use `SELECT VALUE MAX(c.fieldName) FROM c ...`
    3.  **Filtering**: Use standard `WHERE` clauses. For text fields, use `CONTAINS(c.fieldName, 'value')`. For dates, use string comparison (e.g., `c.event_date = '{today_date}'`).
    4.  **Hybrid Queries**: If the question combines semantic search with filters, create a query that includes both a `WHERE` clause and an `ORDER BY VectorDistance(...)`.
    5.  **Response Format**: ALWAYS return ONLY the raw SQL query string. Do not add any explanations, markdown, or other text. If you cannot generate a query, return "SELECT 'Query generation failed: The question is too complex or ambiguous.'"

    Today's date is {today_date}.
    """
    
    response = oai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )
    
    sql_query = response.choices[0].message.content.strip()
    
    # A simple validation to ensure it's a SELECT query
    if not sql_query.upper().startswith("SELECT"):
        raise ValueError("LLM did not generate a valid SELECT query.")
        
    return sql_query

def interpret_results(user_question: str, sql_result: list) -> str:
    """Uses an LLM to convert a raw SQL result into a natural language answer."""
    
    result_str = json.dumps(sql_result, indent=2)
    system_prompt = f"""
    You are an AI assistant. The user asked the following question:
    "{user_question}"

    A database query was executed, and it returned this JSON result:
    {result_str}

    Your task is to interpret this result and provide a friendly, natural language answer to the user's original question.
    If the result is a single value (like a count or an average), state it clearly.
    If the result is a list of items, summarize them briefly.
    If the result is empty, state that no data was found.
    """
    
    response = oai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content

# --- 3. Streamlit User Interface ---

st.set_page_config(page_title="GDELT Intelligent Assistant", layout="wide")
st.title("🤖 GDELT Intelligent Query Assistant")

# Define the schema for the LLM
container_schema = {
    "id": "string (unique identifier)",
    "event_date": "string (YYYY-MM-DD)",
    "actor1_name": "string",
    "actor2_name": "string",
    "avg_tone": "number",
    "goldstein_scale": "number",
    "num_articles": "number",
    "themes": "string (semicolon-separated)",
    "locations": "string (semicolon-separated)",
    "persons": "string (semicolon-separated)",
    "organizations": "string (semicolon-separated)",
    "content": "string (summary for vector search)",
    "contentVector": "array of numbers (for semantic search)"
}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything! For example: 'How many events happened today?' or 'Tell me about climate change protests.'"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing question and querying database..."):
            try:
                # 1. Generate the SQL query from the user's question
                sql_query = generate_cosmos_sql(prompt, container_schema)
                st.sidebar.subheader("Generated SQL Query")
                st.sidebar.code(sql_query, language="sql")

                params = []
                # If it's a vector search query, we need to generate the vector embedding
                if "VectorDistance" in sql_query:
                    query_vector = oai_client.embeddings.create(input=[prompt], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding
                    params.append({"name": "@query_vector", "value": query_vector})

                # 2. Execute the query
                results = list(container.query_items(sql_query, parameters=params, enable_cross_partition_query=True))

                # 3. Interpret the results and generate a natural language answer
                final_answer = interpret_results(prompt, results)
                st.markdown(final_answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                final_answer = "Sorry, I encountered an error while processing your request."

    st.session_state.messages.append({"role": "assistant", "content": final_answer})