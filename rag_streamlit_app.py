import streamlit as st
import os
import json
from openai import AzureOpenAI
from azure.cosmos import CosmosClient

# --- 1. Configuration and Initialization ---

# Load credentials from environment variables
# These must be set in your App Service configuration or local environment
try:
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] # For GPT-4/3.5
    
    COSMOS_DB_ENDPOINT = os.environ["COSMOS_DB_ENDPOINT"]
    COSMOS_DB_KEY = os.environ["COSMOS_DB_KEY"]
    COSMOS_DB_DATABASE_NAME = os.environ["COSMOS_DB_DATABASE_NAME"]
    COSMOS_DB_COLLECTION_NAME = os.environ["COSMOS_DB_COLLECTION_NAME"]
except KeyError as e:
    st.error(f"FATAL ERROR: Environment variable {e} is not set. Please configure it before running the app.")
    st.stop()

# Initialize clients using Streamlit's caching for performance
@st.cache_resource
def get_clients():
    """Initializes and returns the Azure OpenAI and Cosmos DB clients."""
    oai_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY, api_version="2023-05-15", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, credential=COSMOS_DB_KEY)
    database = cosmos_client.get_database_client(COSMOS_DB_DATABASE_NAME)
    container = database.get_container_client(COSMOS_DB_COLLECTION_NAME)
    return oai_client, container

oai_client, container = get_clients()

# --- 2. Streamlit User Interface ---

st.set_page_config(page_title="GDELT RAG Assistant", layout="wide")
st.title("🤖 GDELT Hybrid RAG Assistant")

# Sidebar for hybrid query filters
st.sidebar.header("Hybrid Search Filters")
goldstein_filter = st.sidebar.slider("Minimum Goldstein Scale", -10.0, 10.0, -10.0, 0.5)
tone_filter = st.sidebar.slider("Minimum Average Tone", -10.0, 10.0, -10.0, 0.5)
actor_filter = st.sidebar.text_input("Actor Name (contains)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you analyze GDELT events today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. RAG Logic and Chat Interaction ---

if prompt := st.chat_input("Ask a question, e.g., 'What are the latest diplomatic events involving Russia?'"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Vectorize the user's query
            query_vector = oai_client.embeddings.create(input=[prompt], model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT).data[0].embedding

            # 2. Construct the hybrid query for Cosmos DB
            query_text = "SELECT TOP 5 c.id, c.content, c.source_url FROM c"
            params = [{"name": "@query_vector", "value": query_vector}]
            filter_clauses = []

            # Add metadata filters based on UI input
            if goldstein_filter > -10.0:
                filter_clauses.append("c.goldstein_scale >= @goldstein_scale")
                params.append({"name": "@goldstein_scale", "value": goldstein_filter})
            
            if tone_filter > -10.0:
                filter_clauses.append("c.avg_tone >= @avg_tone")
                params.append({"name": "@avg_tone", "value": tone_filter})

            if actor_filter:
                filter_clauses.append("(CONTAINS(c.actor1_name, @actor_name) OR CONTAINS(c.actor2_name, @actor_name))")
                params.append({"name": "@actor_name", "value": actor_filter})

            if filter_clauses:
                query_text += " WHERE " + " AND ".join(filter_clauses)
            
            # Add the vector search ordering
            query_text += " ORDER BY VectorDistance(c.contentVector, @query_vector)"

            # 3. Execute the query
            try:
                results = list(container.query_items(query_text, parameters=params, enable_cross_partition_query=True))
            except Exception as e:
                st.error(f"Error querying Cosmos DB: {e}")
                st.stop()

            # 4. Generate the final answer using the retrieved context
            if not results:
                final_answer = "I couldn't find any relevant events matching your criteria. Please try adjusting the filters or your question."
                st.write(final_answer)
            else:
                context = "\n\n---\n\n".join([f"Source: {item['source_url']}\nContent: {item['content']}" for item in results])
                system_prompt = f"""
                You are an expert AI assistant for the GDELT dataset.
                Answer the user's question based ONLY on the following context.
                Cite the sources for your answer by referencing the source URLs.
                If the context doesn't contain the answer, state that clearly.

                Context:
                {context}
                """
                
                completion = oai_client.chat.completions.create(
                    model=AZURE_OPENAI_CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                final_answer = completion.choices[0].message.content
                st.markdown(final_answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
