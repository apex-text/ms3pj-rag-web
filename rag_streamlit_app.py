import streamlit as st
import streamlit.components.v1 as components
import os
import json
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from datetime import datetime, timezone
import logging
import traceback
import prompts

# --- 0. Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
except KeyError as e:
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
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    system_prompt = prompts.get_system_prompt().format(today_date=today_date)
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]
    
    response = oai_client.chat.completions.create(model=AZURE_OPENAI_CHAT_DEPLOYMENT, messages=messages_for_api)
    sql_query = response.choices[0].message.content.strip()
    
    if not sql_query.upper().startswith("SELECT"):
        raise ValueError(f"LLM generated a non-SELECT query: {sql_query}")
    return sql_query

def interpret_results(chat_history: list, sql_result: list) -> str:
    user_question = chat_history[-1]["content"] if chat_history else "the user's question"
    result_str = json.dumps(sql_result, indent=2)
    system_prompt = f"""
    You are an AI assistant. Your task is to interpret a JSON database result and answer the user's question based on it.
    The user's most recent question was: "{user_question}"
    A database query returned this JSON result: {result_str}

    **Your instructions:**
    - IF the user asks for links/sources AND the result has `source_url`, format the answer as a Markdown list of links.
    - ELSE, provide a friendly, natural language summary of the result.
    - If the result is empty, state that no data was found.
    """
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]
    response = oai_client.chat.completions.create(model=AZURE_OPENAI_CHAT_DEPLOYMENT, messages=messages_for_api)
    return response.choices[0].message.content

# --- 3. Streamlit User Interface ---

def render_floating_chat():
    """Renders the chat widget using a stable, state-driven approach."""
    with st.expander("🤖 GDELT 어시스턴트", expanded=True):
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "무엇이든 물어보세요! 예: '오늘 발생한 이벤트는 몇 개인가요?'"}]

        # Display chat messages from history
        message_container = st.container()
        with message_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Handle new user input
        if prompt := st.chat_input("Your question..."):
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Process the request and get a response
            with st.spinner("답변을 생성하는 중..."):
                try:
                    sql_query = generate_cosmos_sql(st.session_state.messages)
                    st.sidebar.subheader("Last Generated SQL Query")
                    st.sidebar.code(sql_query, language="sql")

                    params = []
                    if "VectorDistance" in sql_query:
                        query_vector = oai_client.embeddings.create(input=[prompt], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding
                        params.append({"name": "@query_vector", "value": query_vector})

                    results = list(container.query_items(sql_query, parameters=params, enable_cross_partition_query=True))
                    final_answer = interpret_results(st.session_state.messages, results)

                except Exception as e:
                    logging.exception("An error occurred during query processing.")
                    final_answer = "죄송합니다, 요청을 처리하는 중에 오류가 발생했습니다."
                    # Optionally log to browser console for debugging
                    # tb_str = traceback.format_exc()
                    # components.html(f'<script>console.error(`{tb_str}`)</script>', height=0)

            # Add assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # The script will now automatically re-render upon state change without a full rerun.

# --- Main App Layout ---
st.set_page_config(page_title="GDELT Dashboard", layout="wide", initial_sidebar_state="collapsed")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("assets/style.css")

@st.cache_resource
def display_powerbi_dashboard():
    """Caches the Power BI iframe and renders it with st.markdown to prevent reloads."""
    POWERBI_URL = "https://app.powerbi.com/reportEmbed?reportId=60b4e583-90df-4d0a-8719-81f5a29eccd1&autoAuth=true&ctid=8f91900e-dfe5-480a-9a92-56239f989454"
    iframe_html = f'<div class="iframe-container"><iframe title="대시보드" src="{POWERBI_URL}" frameborder="0" allowFullScreen="true"></iframe></div>'
    st.markdown(iframe_html, unsafe_allow_html=True)

display_powerbi_dashboard()
render_floating_chat()