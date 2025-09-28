import streamlit as st
import streamlit.components.v1 as components
import os
import json
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
import pandas as pd
from datetime import datetime, timezone
import logging
import traceback
import prompts # Import the new prompts module

# --- 0. Logging Configuration ---
# Logs will be written to stdout, which Azure Web App captures in Log Stream.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- 1. Configuration and Initialization ---
logging.info("Streamlit app starting up...")

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
    logging.info("Successfully loaded all environment variables.")
except KeyError as e:
    logging.critical(f"FATAL ERROR: Environment variable {e} is not set.")
    st.error(f"FATAL ERROR: Environment variable {e} is not set.")
    st.stop()

# Initialize clients
@st.cache_resource
def get_clients():
    logging.info("Initializing OpenAI and Cosmos DB clients...")
    oai_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, credential=COSMOS_DB_KEY)
    database = cosmos_client.get_database_client(COSMOS_DB_DATABASE_NAME)
    container = database.get_container_client(COSMOS_DB_COLLECTION_NAME)
    logging.info("Clients initialized and cached.")
    return oai_client, container

oai_client, container = get_clients()

def generate_cosmos_sql(chat_history: list) -> str:
    """Uses an LLM to generate a Cosmos DB SQL query from a natural language question, using conversation history for context."""
    
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    system_prompt = prompts.get_system_prompt().format(today_date=today_date)
    
    # Combine the system prompt with the last 10 messages from the chat history
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]
    
    logging.info(f"Generating SQL query with OpenAI using the last {len(chat_history[-10:])} messages for context.")
    response = oai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages_for_api
    )
    
    sql_query = response.choices[0].message.content.strip()
    logging.info(f"Generated SQL query: {sql_query}")
    
    if not sql_query.upper().startswith("SELECT"):
        error_message = f"LLM did not generate a valid SELECT query. Instead, it returned: {sql_query}"
        logging.error(error_message)
        raise ValueError(error_message)
        
    return sql_query

def interpret_results(chat_history: list, sql_result: list) -> str:
    """Uses an LLM to convert a raw SQL result into a natural language answer, using conversation history for context."""
    
    # The user's last question is the last message in the history
    user_question = chat_history[-1]["content"] if chat_history else "the user's question"
    
    result_str = json.dumps(sql_result, indent=2)
    system_prompt = f"""
    You are an AI assistant. The user's most recent question was:
    "{user_question}"

    A database query was executed, and it returned this JSON result:
    {result_str}

    Based on the result and the ongoing conversation, provide a friendly, natural language answer to the user's question.
    If the result is a single value (like a count), state it clearly.
    If the result is a list of items, summarize them briefly.
    If the result is empty, state that no data was found that matches their request.
    Keep the answer concise and relevant to the last question.
    """
    
    # Combine system prompt with conversation history for a more natural, context-aware summary
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]

    logging.info("Interpreting SQL results with OpenAI...")
    response = oai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages_for_api
    )
    final_answer = response.choices[0].message.content
    logging.info("Successfully interpreted results.")
    return final_answer

# --- 3. Streamlit User Interface ---

def log_to_browser(message):
    """Injects a script to log a message to the browser's console."""
    # Escape characters for safe injection into a JavaScript template literal
    escaped_message = message.replace('\\', '\\\\').replace('`', '\`').replace('"', '\"').replace("'", "\'\'")
    components.html(f'<script>console.error(`{escaped_message}`)</script>', height=0)

def render_floating_chat():
    """Renders the floating chat widget with a fixed input bar."""

    with st.expander("🤖 GDELT Assistant", expanded=True):
        
        # Create a container for the chat history that will be scrollable
        message_container = st.container()

        with message_container:
            # Initialize and display chat history
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Ask me anything! For example: 'How many events happened today?' or 'Tell me about climate change protests.'"}]
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Handle new user input (this will be rendered below the scrollable container)
        if prompt := st.chat_input("Your question..."):
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process and get assistant response
            try:
                with st.spinner("Analyzing question and querying database..."):
                    # 1. Generate SQL query
                    sql_query = generate_cosmos_sql(st.session_state.messages)
                    st.sidebar.subheader("Last Generated SQL Query")
                    st.sidebar.code(sql_query, language="sql")

                    # 2. Generate vector embedding if needed
                    params = []
                    if "VectorDistance" in sql_query:
                        logging.info("VectorDistance detected, generating query vector embedding...")
                        query_vector = oai_client.embeddings.create(input=[prompt], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding
                        params.append({"name": "@query_vector", "value": query_vector})
                        logging.info("Query vector generated.")

                    # 3. Execute query
                    logging.info("Executing query against Cosmos DB...")
                    results = list(container.query_items(sql_query, parameters=params, enable_cross_partition_query=True))
                    logging.info(f"Query returned {len(results)} results.")

                    # 4. Interpret results
                    final_answer = interpret_results(st.session_state.messages, results)

            except Exception as e:
                logging.exception("An error occurred during query processing.")
                final_answer = "Sorry, I encountered an error while processing your request."
                
                if hasattr(e, 'body') and e.body:
                    error_details = e.body.get('message', json.dumps(e.body))
                else:
                    tb_str = traceback.format_exc()
                    error_details = tb_str
                
                log_to_browser(f"RAG App Error: {e}\n{error_details}")
                # Display error in the main UI as well, but no need for a separate st.error here
                # as the final_answer will be displayed in the chat.

            # Add assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # The script will now automatically rerun from the chat_input interaction,
            # displaying the new messages without a forced rerun.

# --- Main App Layout ---
st.set_page_config(page_title="GDELT Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Function to load and inject an external CSS file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 1. Inject CSS from external file
load_css("assets/style.css")

# 2. Display the Power BI dashboard in fullscreen
st.sidebar.title("SQL Query")
POWERBI_URL = "https://app.powerbi.com/reportEmbed?reportId=60b4e583-90df-4d0a-8719-81f5a29eccd1&autoAuth=true&ctid=8f91900e-dfe5-480a-9a92-56239f989454"
st.markdown(f'<div class="iframe-container"><iframe title="대시보드" src="{POWERBI_URL}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)

# 3. Render the floating chat widget
render_floating_chat()