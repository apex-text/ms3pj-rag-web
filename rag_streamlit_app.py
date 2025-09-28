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
    today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    system_prompt = prompts.get_system_prompt().format(today_date=today_date)
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]
    
    logging.info(f"Generating SQL query with OpenAI...")
    response = oai_client.chat.completions.create(model=AZURE_OPENAI_CHAT_DEPLOYMENT, messages=messages_for_api)
    sql_query = response.choices[0].message.content.strip()
    logging.info(f"Generated SQL query: {sql_query}")
    
    if not sql_query.upper().startswith("SELECT"):
        raise ValueError(f"LLM did not generate a valid SELECT query. Instead, it returned: {sql_query}")
        
    return sql_query

def interpret_results(chat_history: list, sql_result: list) -> str:
    user_question = chat_history[-1]["content"] if chat_history else "the user's question"
    result_str = json.dumps(sql_result, indent=2)
    system_prompt = f"""
    You are an AI assistant. Your task is to interpret a JSON database result and answer the user's question based on it.
    The user's most recent question was: "{user_question}"
    A database query returned this JSON result: {result_str}

    **Your instructions:**
    1.  Analyze the user's question. Does it contain words like "link", "source", "URL", "출처", "링크", "소스"?
    2.  Check the JSON result. Does it contain a `source_url` field?
    3.  Generate the answer based on the following logic:
        - IF the user is asking for links AND the JSON result contains valid `source_url`s, your primary goal is to provide those links. Format the answer as a Markdown list. For each item, use the `content` as the link text and `source_url` as the URL. For example: `* [Event summary text](http://example.com/news_article)`
        - ELSE, provide a friendly, natural language summary of the JSON result.
            - If the result is a single value (like a count), state it clearly.
            - If the result is a list of items, summarize them briefly.
            - If the result is empty, state that no data was found.
    Keep the answer concise and directly related to the user's question.
    """
    messages_for_api = [{"role": "system", "content": system_prompt}] + chat_history[-10:]

    logging.info("Interpreting SQL results with OpenAI...")
    response = oai_client.chat.completions.create(model=AZURE_OPENAI_CHAT_DEPLOYMENT, messages=messages_for_api)
    final_answer = response.choices[0].message.content
    logging.info("Successfully interpreted results.")
    return final_answer

# --- 3. Streamlit User Interface ---

def log_to_browser(message):
    escaped_message = message.replace('\\', '\\\\').replace('`', '\`').replace('"', '\"').replace("'", "\'\'")
    components.html(f'<script>console.error(`{escaped_message}`)</script>', height=0)

def render_floating_chat():
    """Renders the floating chat widget with a stable, streaming-like UI."""
    with st.expander("🤖 GDELT 어시스턴트", expanded=True):
        
        # This container holds the scrollable chat history
        message_container = st.container()
        
        # Initialize or display chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "무엇이든 물어보세요! 예: '오늘 발생한 이벤트는 몇 개인가요?' 또는 '기후 관련 갈등에 대해 알려주세요.'"}]
        
        for message in st.session_state.messages:
            with message_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # The chat input is outside the container, at the bottom of the expander
        if prompt := st.chat_input("Your question..."):
            # Add user message to state and display it immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with message_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Show a spinner in the assistant's placeholder while processing
            with message_container:
                with st.chat_message("assistant"):
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
                            tb_str = traceback.format_exc()
                            log_to_browser(f"RAG App Error: {e}\n{tb_str}")
                        
                        # Replace spinner with the final answer
                        st.markdown(final_answer)
            
            # Add assistant's response to the session state for the next rerun
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

# --- Main App Layout ---
st.set_page_config(page_title="GDELT Dashboard", layout="wide", initial_sidebar_state="collapsed")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("assets/style.css")

@st.cache_resource
def display_powerbi_dashboard():
    """Caches the Power BI iframe to prevent it from reloading on every script rerun."""
    POWERBI_URL = "https://app.powerbi.com/reportEmbed?reportId=60b4e583-90df-4d0a-8719-81f5a29eccd1&autoAuth=true&ctid=8f91900e-dfe5-480a-9a92-56239f989454"
    iframe_html = f'<div class="iframe-container"><iframe title="대시보드" src="{POWERBI_URL}" frameborder="0" allowFullScreen="true"></iframe></div>'
    # Use components.html for more stable rendering of static HTML
    components.html(iframe_html, height=0)

display_powerbi_dashboard()
render_floating_chat()
