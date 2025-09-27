import streamlit as st
import streamlit.components.v1 as components
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

def render_floating_chat():
    """Renders the floating chat widget using a styled st.expander."""

    # Define the schema for the LLM inside the function or pass it as an argument
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

    with st.expander("🤖 GDELT Assistant", expanded=False):
        # This container holds the scrollable chat history
        message_container = st.container()

        # This container holds the input box
        input_container = st.container()

        # Initialize or get chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ask me anything! For example: 'How many events happened today?' or 'Tell me about climate change protests.'"}]

        # Display chat history in the message container
        with message_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Handle new user input in the input container
        with input_container:
            if prompt := st.chat_input("Your question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Analyzing question and querying database..."):
                    try:
                        # 1. Generate SQL query
                        sql_query = generate_cosmos_sql(prompt, container_schema)
                        
                        # Display generated query in the main sidebar for debugging
                        st.sidebar.subheader("Last Generated SQL Query")
                        st.sidebar.code(sql_query, language="sql")

                        params = []
                        # 2. Generate vector embedding if needed
                        if "VectorDistance" in sql_query:
                            query_vector = oai_client.embeddings.create(input=[prompt], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding
                            params.append({"name": "@query_vector", "value": query_vector})

                        # 3. Execute query
                        results = list(container.query_items(sql_query, parameters=params, enable_cross_partition_query=True))

                        # 4. Interpret results for a natural language answer
                        final_answer = interpret_results(prompt, results)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        final_answer = "Sorry, I encountered an error while processing your request."

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.rerun()

# --- Main App Layout ---
st.set_page_config(page_title="GDELT Dashboard", layout="wide", initial_sidebar_state="collapsed")

# 1. Inject CSS to style the st.expander as a floating widget
st.markdown("""
<style>
    /* Main container for the floating expander */
    div[data-testid="stExpander"] {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 450px;
        max-width: 90vw;
        z-index: 1000;
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Expander header styling */
    div[data-testid="stExpander"] > div[role="button"] {
        background-color: #007bff;
        color: white;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
    }

    /* The direct content area of the expander */
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div:nth-of-type(1) {
        max-height: 65vh; /* Set a MAX height for the content area */
        display: flex;
        overflow-y: auto;  /* Make this container scrollable */
        flex-direction: column;
    }

    /* Target the st.container() holding the messages (first child) */
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div:nth-of-type(1) > div:nth-of-type(1) {
        flex-grow: 1;      /* Allow the message container to grow */
        padding-right: 10px; /* Add some padding for the scrollbar */
    }

    /* Target the st.container() for the chat input (second child) */
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div:nth-of-type(1) > div:nth-of-type(2) {
        flex-shrink: 0; /* Prevent the input container from shrinking */
    }
</style>
""", unsafe_allow_html=True)

# 2. Display the Power BI dashboard
st.sidebar.title("SQL Query")
st.header("Power BI Dashboard")
POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiNDJlN2RmMDAtZDg5Ni00MjA3LThiZjMtMDQyZGQ1NDU3Njg2IiwidCI6IjhmOTE5MDBlLWRmZTUtNDgwYS05YTkyLTU2MjM5Zjk4OTQ1NCJ9"
components.iframe(POWERBI_URL, height=700, scrolling=True)


# IMPORTANT: Replace these URLs with the public or embeddable URLs of your dashboards
 # Placeholder URL


# 3. Render the floating chat widget
render_floating_chat()