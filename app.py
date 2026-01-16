# -*- coding: utf-8 -*-
"""
This module implements a Streamlit web application for searching research papers.

The application serves as a retrieval-augmented generation (RAG) tool. It allows
users to query a pre-built vector database of research documents. Key features
include:
- On-demand downloading and caching of vector databases from a remote source.
- A user interface to select different databases (e.g., full corpus vs. journals).
- AI-powered query enhancement to improve search relevance.
- AI-powered summarization of search results with citations.
- A clear, interactive display of search results, including document metadata
  and content snippets.
- User reporting mechanism for data quality issues (Document-level).
- Results grouped by parent document.
- An "Author Explorer" tab to find top authors by subject.
- A "Query Analyzer" tab to test consultation questions against the database.

The application relies on Streamlit for the UI, LangChain for vector store
management, Hugging Face for embedding models, and OpenAI for the LLM-powered
features.
"""

import streamlit as st
import os
import datetime
from langchain_qdrant import Qdrant
from qdrant_client.http.models import Filter, FieldCondition, Range
from langchain_huggingface import HuggingFaceEmbeddings
import openai
import tiktoken
from sentence_transformers import CrossEncoder
import collections  # Added for counting authors
import pandas as pd  # Added for displaying author results
import altair as alt # Added for custom bar chart
import json         # Added for parsing LLM-generated question lists
import qdrant_client
import streamlit as st
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
import gspread
from oauth2client.service_account import ServiceAccountCredentials

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("BAAI/bge-reranker-large")

def rerank_results(query, docs, reranker_model, top_k=10):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker_model.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in reranked[:top_k]]
    return reranked_docs

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io" 

COLLECTION_FULL = "full_papers" 
COLLECTION_JOURNAL = "journal_papers" 
COLLECTION_EDRC = "edrc_papers"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
REPORT_SHEET_NAME = "RAG Data Reports" 
MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}

@st.cache_resource
def load_embedding_model():
    """Loads and caches the sentence embedding model from Hugging Face.

    This function initializes the `HuggingFaceEmbeddings` model specified by
    the `EMBEDDING_MODEL_NAME` constant. The `@st.cache_resource` decorator
    ensures the model is loaded only once and cached for subsequent runs.

    Returns:
        langchain_huggingface.HuggingFaceEmbeddings: The loaded embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_full_store(_embeddings, _url, _api_key):
    """Loads and caches the FULL vector store with explicit client creation."""
    
    # 1. Manually create the client. This ensures we aren't using a hidden default.
    client = QdrantClient(
        url=_url, 
        api_key=_api_key,
        prefer_grpc=True # Optional: often more stable for cloud
    )
    
    # 2. Use QdrantVectorStore (new class) instead of Qdrant (legacy alias)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_FULL,
        embedding=_embeddings,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )

@st.cache_resource
def load_journal_store(_embeddings, _url, _api_key):
    """Loads and caches the JOURNAL vector store."""
    client = QdrantClient(
        url=_url, 
        api_key=_api_key,
        prefer_grpc=True
    )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_JOURNAL,
        embedding=_embeddings,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )

@st.cache_resource
def load_edrc_store(_embeddings, _url, _api_key):
    """Loads and caches the EDRC vector store."""
    client = QdrantClient(
        url=_url, 
        api_key=_api_key,
        prefer_grpc=True
    )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_EDRC,
        embedding=_embeddings,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )

@st.cache_data
def count_tokens(text: str, model: str = "gpt-5-nano") -> int:
    """Counts the number of tokens in a text string for a given model.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # If tiktoken does not have a direct mapping for the model,
        # we fall back to a general-purpose tokenizer.
        st.warning(f"Tokenizer for model '{model}' not found. Using 'cl100k_base' as a fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def display_token_usage(token_info, model_name, title):
    """Displays token usage and estimated cost in a Streamlit expander."""
    input_tokens = token_info.get('input_tokens', 0)
    output_tokens = token_info.get('output_tokens', 0)
    total_tokens = token_info.get('total_tokens', 0)

    # Calculate cost based on the selected model
    model_pricing = MODEL_COSTS.get(model_name, {"input": 0, "output": 0})
    input_cost_per_mil = model_pricing.get("input", 0)
    output_cost_per_mil = model_pricing.get("output", 0)
    cost = (input_tokens * input_cost_per_mil / 1_000_000) + (output_tokens * output_cost_per_mil / 1_000_000)
    co2 = output_tokens * 0.159/50 # CO2 in g per token x tokens

    with st.expander(f"Token Usage Details: {title}"):
        st.markdown(f"- **Input Tokens:** `{input_tokens}`")
        st.markdown(f"- **Output Tokens:** `{output_tokens}`")
        st.markdown(f"- **Total Tokens:** `{total_tokens}`")
        st.markdown(f"- **Estimated Cost:** `${cost:.6f}` (Model: `{model_name}`)")
        st.markdown(f"- **Estimated CO2:** `{co2:.2f} g`")

def submit_report_to_sheets(doc_metadata, chunk_content, reason):
    """Submits a data quality report to a Google Sheet.

    Uses service account credentials stored in Streamlit secrets to authenticate
    and append a new row to the configured Google Sheet.

    Args:
        doc_metadata (dict): Metadata of the document chunk being reported.
        chunk_content (str): The actual text content of the chunk.
        reason (str): The user-provided reason for the report.

    Returns:
        bool: True if submission was successful, False otherwise.
    """
    try:
        # Check if secrets exist before attempting connection
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud credentials not found in Streamlit secrets.")
            return False

        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], SCOPES)
        client = gspread.authorize(creds)
        
        # Open the spreadsheet by name.
        sheet = client.open(REPORT_SHEET_NAME).sheet1
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare the row data. Truncate snippet to avoid exceeding cell limits if necessary.
        row = [
            timestamp,
            doc_metadata.get('title', 'N/A'),
            doc_metadata.get('source', 'N/A'),
            str(doc_metadata.get('year', 'N/A')),
            reason,
            chunk_content[:500] + "..." 
        ]
        sheet.append_row(row)
        return True
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet '{REPORT_SHEET_NAME}' not found. Please check the name and ensure the service account has editor access.")
        return False
    except Exception as e:
        st.error(f"Failed to submit report due to an error: {e}")
        return False
        
def log_usage_stats(user_query, collection_name, results_count, enhanced_mode, summary_mode):
    """
    Logs anonymous usage stats to the 'Usage Logs' sheet.
    """
    try:
        # Check for secrets just like you do for reports
        if "gcp_service_account" not in st.secrets:
            return # Fail silently so users aren't disturbed

        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], SCOPES)
        client = gspread.authorize(creds)
        
        # Open the specific worksheet
        sheet = client.open(REPORT_SHEET_NAME).worksheet("Usage Logs")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            timestamp,
            user_query,
            collection_name,
            results_count,
            str(enhanced_mode),
            str(summary_mode)
        ]
        
        sheet.append_row(row)
    except Exception as e:
        # Print to console for admin debugging, but don't show user an error
        print(f"Logging failed: {e}")

def improve_query_with_llm(user_query):
    """Improves a user's query using an LLM for better search results.

    This function sends the user's query to an OpenAI model with a prompt
    that asks it to rephrase the query for a vector database search, focusing
    on academic keywords.

    Args:
        user_query (str): The original query entered by the user.

    Returns:
        tuple[str, dict | None]: A tuple containing the enhanced query and a
        dictionary with token usage, or None if the API call fails.
    """
    try:
        prompt = f"""
        You are an expert at query optimization for vector databases. Your task is to rephrase the following user's question to be more effective for a semantic search.
        Focus on using keywords and concepts found in academic research papers. Do not answer the question. Only provide the single, rephrased query.
        
        User's question: "{user_query}"
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful query optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        improved_query = response.choices[0].message.content.strip()
        
        usage = response.usage
        token_info = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        
        return improved_query.strip('"'), token_info
    except Exception as e:
        st.warning(f"Could not improve query due to an API error: {e}. Using the original query.")
        return user_query, None


def extract_questions_with_llm(consultation_text):
    """
    Uses an LLM to extract a list of questions from a block of text.

    Args:
        consultation_text (str): The full text of the consultation.

    Returns:
        tuple[list[str] | None, dict | None]: A tuple containing:
            - A list of extracted question strings (or None on failure).
            - A dictionary with token usage (or None on API failure).
    """
    try:
        prompt = f"""
        You are a text-processing bot. Your task is to read the following text and extract a list of all explicit and implicit questions.
        - Ignore headings, numbering, and introductory text.
        - Focus only on the questions themselves.
        - Return *only* a valid JSON list of strings. Do not include any preamble, markdown, or other text.
        
        Example output:
        ["What is the national vision?", "Why do you disagree?", "How can ATE support local authorities?"]
        
        --- Text to Analyze ---
        {consultation_text}
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a text-processing bot that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"} # Use JSON mode
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Capture token usage
        usage = response.usage
        token_info = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        
        # The model is asked for a JSON list, but might return a JSON object 
        # like {"questions": ["..."]}. We need to handle both.
        try:
            # Try to parse the whole string as a list
            data = json.loads(response_text)
            if isinstance(data, list):
                return data, token_info
            # If it's a dictionary, look for a key that contains a list
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value, token_info # Return the first list found
            
            st.error(f"LLM returned valid JSON, but not in the expected format (list of strings or object with a list): {response_text}")
            return None, token_info # Return token info even on parse error

        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON from LLM response: {response_text}")
            return None, token_info # Return token info even on decode error
        
    except Exception as e:
        st.warning(f"Could not extract questions due to an API error: {e}.")
        return None, None # API error, no token info


def summarize_results_with_llm(user_query, _search_results, model="gpt-5-nano", max_completion_tokens=10000):
    """Generates an AI-powered summary of search results with citations.

    This function constructs a detailed prompt containing the user's original
    question and snippets from the search results. It asks an OpenAI model to
    synthesize an answer based *only* on the provided context and to cite
    the sources for each piece of information.

    Args:
        user_query (str): The original query entered by the user.
        _search_results (list[langchain.docstore.document.Document]): A list of
            the top documents returned from the vector search.
        model (str): The name of the OpenAI model to use.
        max_completion_tokens (int): The maximum number of tokens to generate for the summary.

    Returns:
        tuple[str | None, dict | None]: A tuple containing the summary and a
        dictionary with token usage, or None if the API call fails.
    """
    try:
        context_snippets = []
        for i, doc in enumerate(_search_results):
            title = doc.metadata.get('title', 'No Title Found')
            citation_marker = f"[{i+1}]"
            context_snippets.append(f"--- Source {citation_marker} ---\nTitle: {title}\nSnippet: {doc.page_content}\n---")

        full_context = "\n\n".join(context_snippets)

        prompt = f"""
        You are a research assistant. Your task is to synthesize the provided document snippets to answer the user's question. Follow these rules strictly:
        ## Core Instructions

            1. Source Adherence: Base your entire response exclusively on the information within the "Document Snippets." Do not introduce any outside knowledge or assumptions.

            2. Synthesize, Don't Just List: Weave information from the snippets into a cohesive summary. Instead of quoting directly, integrate and combine related points from different sources to fully answer the user's question.

            3. Precise In-line Citations: You must cite every claim or piece of information. Place citation markers directly after the relevant sentence or clause.

            4. For a single source, use [1].

            5. For multiple sources supporting one statement, combine them like [1, 3].

            6. Handle Insufficient Information: If the provided snippets do not contain enough information to answer the question, state this clearly.

            7. You must use a paragraph format, grouping information by theme.

            8. References Section: After your summary, add a ## References section. List all the provided document snippets numerically in bullet point order, corresponding to your in-line citations.

        User's question: "{user_query}"

        --- Document Snippets ---
        {full_context}

        --- Synthesized Answer ---
        [Your summary with citations goes here]

        **References**
        [Your numbered list of references goes here, e.g., "[1] Title of the first paper."]
        """
        
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that provides citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=1, 
            max_completion_tokens=max_completion_tokens
        )
        summary = response.choices[0].message.content.strip()
        usage = response.usage
        token_info = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        
        return summary, token_info
    except Exception as e:
        st.error(f"Could not generate summary due to an API error: {e}")
        return None, None

def group_results(results):
    """
    Groups document chunks by their source document.
    
    This ensures that if multiple chunks from the same paper are returned,
    they are displayed together under a single document heading, rather than
    as separate, repetitive entries in the results list.
    """
    grouped = {}
    ordered_groups = []
    for doc in results:
        # Use 'source' as the unique key, fallback to 'title' if missing
        key = doc.metadata.get('source') or doc.metadata.get('title') or "unknown_doc"
        
        if key not in grouped:
            # Create new group entry
            group_data = {
                "metadata": doc.metadata, # Use metadata from the first (highest ranked) chunk
                "chunks": []
            }
            grouped[key] = group_data
            ordered_groups.append(group_data)
        
        # Add chunk to existing group
        grouped[key]["chunks"].append(doc)
    
    return ordered_groups

def main():
    """Defines the main function to run the Streamlit application.

    This function sets up the Streamlit page configuration, handles the initial
    database checks and downloads, and builds the user interface. It manages
    the application's state and logic for handling user input and search
    execution.
    """
    st.set_page_config(page_title="Research Paper Search", page_icon="📚", layout="wide")

    if 'final_query' not in st.session_state:
        st.session_state.final_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'original_query' not in st.session_state:
        st.session_state.original_query = ""
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False
    if 'summary_content' not in st.session_state:
        st.session_state.summary_content = None
    if 'summary_token_info' not in st.session_state:
        st.session_state.summary_token_info = None
    if 'selected_collection' not in st.session_state:
        st.session_state.selected_collection = COLLECTION_FULL # Default to the first option

    # Initialize form widget states if they don't exist
    if "user_query_input" not in st.session_state:
        st.session_state.user_query_input = ""
    if "k_results_input" not in st.session_state:
        st.session_state.k_results_input = 10
    if "enhanced_search_toggle" not in st.session_state:
        st.session_state.enhanced_search_toggle = True
    if "summary_toggle" not in st.session_state:
        st.session_state.summary_toggle = True
    if "reranker_toggle" not in st.session_state:
        st.session_state.reranker_toggle = True
    if "start_date_input" not in st.session_state:
        st.session_state.start_date_input = 2015
    if "end_date_input" not in st.session_state:
        st.session_state.end_date_input = 2024
    if "date_filter_toggle" not in st.session_state:
        st.session_state.date_filter_toggle = False

    st.title("📚 Research Paper Search")
    st.write("Ask a question about your documents, and the app will find the most relevant information.")
    
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    
    api_key_present = bool(openai_api_key)

    if not openai_api_key:
        st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. AI-powered features will be disabled.", icon="⚠️")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect to database.", icon="🚨")
        st.stop()    
        
    DB_OPTIONS = {
        "Full Database": COLLECTION_FULL,
        "Journal Articles Only": COLLECTION_JOURNAL,
        "EDRC Only": COLLECTION_EDRC,
    }
    
    # --- Load Models and Vector Store (needed for both tabs) ---
    try:
        embeddings = load_embedding_model()
        
        # We still need to load the *selected* store for the main search tab
        # The author tab will load the full store independently if needed
        
        try:
            current_collection_index = list(DB_OPTIONS.values()).index(st.session_state.selected_collection)
        except ValueError:
            current_collection_index = 0 # Default to first item if state is invalid

        # Store the db_choice label for use in the main tab caption
        db_choice_label = list(DB_OPTIONS.keys())[current_collection_index]
        selected_collection_name = st.session_state.selected_collection
        
        if selected_collection_name == COLLECTION_FULL:
            vector_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
        elif selected_collection_name == COLLECTION_JOURNAL:
            vector_store = load_journal_store(embeddings, QDRANT_URL, qdrant_api_key)
        else:
            vector_store = load_edrc_store(embeddings, QDRANT_URL, qdrant_api_key)
        
    except Exception as e:
        st.error(f"An error occurred while loading the models or database: {e}")
        st.stop()

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["📚 Document Search", "🧑‍🔬 Author Explorer", "🔎 Query Analyzer"])

    with tab1:
        # --- Database Selection ---
        db_choice = st.radio(
            "Select database to search:",
            options=DB_OPTIONS.keys(),
            horizontal=True,
            index=current_collection_index,  # Use the index from session state
            key="db_selection_radio"
        )
        
        selected_collection_name_from_radio = DB_OPTIONS[db_choice]

        # If the user selected a new database, update the state and clear old results
        if selected_collection_name_from_radio != st.session_state.selected_collection:
            st.session_state.selected_collection = selected_collection_name_from_radio
            
            # Clear all previous search and summary state
            st.session_state.search_results = None
            st.session_state.final_query = ""
            st.session_state.original_query = ""
            st.session_state.summary_generated = False
            st.session_state.summary_content = None
            st.session_state.summary_token_info = None
            
            # Reset the form inputs using their keys
            st.session_state.user_query_input = "" 
            st.session_state.k_results_input = 10
            st.session_state.enhanced_search_toggle = True
            st.session_state.summary_toggle = True
            st.session_state.start_date_input = 2015
            st.session_state.end_date_input = 2024
            st.session_state.date_filter_toggle = False
            
            st.rerun()

        available_models = ["gpt-5-nano", "gpt-4o-mini", "gpt-5-mini"]
        selected_model = st.selectbox(
            "Select AI Model for Summary:",
            options=available_models,
            index=0,
            help="Choose the model for AI summarization. Query enhancement is fixed to gpt-4o-mini for efficiency."
        )

        try:
            count_result = vector_store.client.count(
                collection_name=st.session_state.selected_collection, 
                exact=True
            )
            st.caption(f"ℹ️ `{db_choice_label}` (collection: `{st.session_state.selected_collection}`) loaded with {count_result.count} documents.")
        
        except Exception as e:
            st.error(f"An error occurred while counting documents: {e}")
            st.stop()

        with st.form("search_form"):
            user_query = st.text_input(
                "Ask a question:", 
                placeholder="e.g., What are the effects of policy on renewable energy adoption?",
                key="user_query_input" 
            )
            
            col1, col2, col3 = st.columns([5, 2, 3])
            with col1:
                k_results = st.slider(
                    "Number of results to return:", 
                    min_value=1, max_value=30, 
                    key="k_results_input"
                )
            with col2:
                use_enhanced_search = st.toggle(
                    "AI-Enhanced Search",
                    help="Uses an AI model to rephrase your query.",
                    disabled=not api_key_present,
                    key="enhanced_search_toggle"
                )
            with col3:
                generate_summary = st.toggle(
                    "Generate AI Summary",
                    help="Uses an AI model to summarize the search results.",
                    disabled=not api_key_present,
                    key="summary_toggle"
                )

            # --- Date Range Selection ---
            date_col1, date_col2, date_col3 = st.columns([2, 2, 6])
            with date_col1:
                start_date = st.number_input(
                    "Start Year", min_value=1900, max_value=2100, step=1,
                    key="start_date_input"
                )
            with date_col2:
                end_date = st.number_input(
                    "End Year", min_value=1900, max_value=2100, step=1,
                    key="end_date_input"
                )
            with date_col3:
                use_date_filter = st.checkbox(
                    "Filter by year", 
                    key="date_filter_toggle"
                )

            submitted = st.form_submit_button("Search", type="primary", width='stretch')

        if submitted and user_query:
            st.session_state.search_results = None
            st.session_state.final_query = ""
            st.session_state.original_query = user_query
            st.session_state.summary_generated = False
            st.session_state.summary_content = None
            st.session_state.summary_token_info = None

            if use_enhanced_search and api_key_present:
                with st.spinner("Improving query..."):

                    openai.api_key = st.secrets["OPENAI_API_KEY"]
                    improved_query, token_info = improve_query_with_llm(user_query)

                    if token_info:
                        display_token_usage(token_info, "gpt-4o-mini", "Query Enhancement")
                        st.session_state.final_query = improved_query if improved_query else user_query
                    # If token_info is None, an API error occurred and a warning was already shown.
                    # The app will fall through and show the review box with the original query.
                    elif not st.session_state.final_query:
                        st.session_state.final_query = user_query
        run_final_search = False 

        # --- Display Enhanced Query for Editing ---
        if st.session_state.final_query and not st.session_state.search_results:
            with st.form("final_search_form"):
                st.info("Review and edit the query below, then click 'Run Search'.")
                edited_query = st.text_area("Suggested Query:", value=st.session_state.final_query, height=100)
                
                use_reranker = st.toggle(
                    "Use Reranker (BGE-Large)",
                    help="Re-rank top search results using a cross-encoder model for higher precision.",
                    key="reranker_toggle"
                )
                
                run_final_search = st.form_submit_button("Run Search", type="primary", width='stretch')

            if run_final_search:
                query_to_use = edited_query
                with st.spinner(f"Searching `{db_choice_label}` with final query..."):
                    try:
                        search_kwargs = {"k": k_results}

                        if use_date_filter:
                            search_kwargs["filter"] = Filter(
                                must=[
                                    FieldCondition(
                                        key="metadata.year", 
                                        range=Range(gte=start_date, lte=end_date)
                                    )
                                ]
                            )

                        # Perform initial search
                        initial_results = vector_store.similarity_search(query_to_use, **search_kwargs)
                        
                        if use_reranker:
                            with st.spinner("Re-ranking results..."):
                                reranker_model = load_reranker_model()
                                reranked = rerank_results(query_to_use, initial_results, reranker_model, top_k=k_results)
                                st.session_state.search_results = reranked
                        else:
                            st.session_state.search_results = initial_results
                        log_usage_stats(
                            user_query=query_to_use,
                            collection_name=st.session_state.selected_collection,
                            results_count=len(st.session_state.search_results),
                            enhanced_mode=st.session_state.enhanced_search_toggle,
                            summary_mode=st.session_state.summary_toggle
                        )
                    except Exception as e:
                        st.error(f"An error occurred during the search: {e}")
                st.rerun()

        if st.session_state.search_results is not None:
            results = st.session_state.search_results

            if st.session_state.summary_generated:
                with st.expander("✨ **AI-Generated Summary**", expanded=True):
                    st.markdown(st.session_state.summary_content)
                display_token_usage(st.session_state.summary_token_info, selected_model, "AI Summary")

            # Show confirmation form if summary is requested but not yet generated
            elif generate_summary and results and api_key_present:
                with st.form("summary_confirmation_form"):
                    st.subheader("Generate AI Summary")
                    st.info("A summary will be generated based on the search results. Review the estimated cost below.")

                    context_text = "\n\n".join([f"--- Source [{i+1}] ---\nTitle: {doc.metadata.get('title', 'No Title Found')}\nSnippet: {doc.page_content}\n---" for i, doc in enumerate(st.session_state.search_results)])
                    est_input_tokens = count_tokens(st.session_state.original_query + context_text, model=selected_model)
                    dynamic_max_tokens = est_input_tokens + 1000

                    # Estimate cost
                    model_pricing = MODEL_COSTS.get(selected_model, {"input": 0, "output": 0})
                    input_cost = (est_input_tokens * model_pricing.get("input", 0)) / 1_000_000
                    # Use dynamic_max_tokens for output cost estimation
                    output_cost = (dynamic_max_tokens * model_pricing.get("output", 0)) / 1_000_000
                    estimated_cost = input_cost + output_cost
                    est_co2 = dynamic_max_tokens * 0.159/50 # CO2 in g per token x tokens


                    st.markdown(f"- **Estimated Input Tokens:** `{est_input_tokens}`")
                    st.markdown(f"- **Max Output Tokens:** `{dynamic_max_tokens}`")
                    st.markdown(f"- **Estimated Maximum Cost:** `${estimated_cost:.4f}`")
                    st.markdown(f"- **Estimated Maximum CO2:** `{est_co2:.2f}` g")

                    proceed_with_summary = st.form_submit_button("Generate Summary", type="primary")

                if proceed_with_summary:
                    with st.spinner("Thinking..."):
                        openai.api_key = st.secrets["OPENAI_API_KEY"]
                        summary, token_info = summarize_results_with_llm(st.session_state.original_query, st.session_state.search_results, model=selected_model, max_completion_tokens=dynamic_max_tokens)

                        if summary and token_info:
                            st.session_state.summary_content = summary
                            st.session_state.summary_token_info = token_info
                            st.session_state.summary_generated = True
                            st.rerun()
                        else:
                            st.warning("The AI summary could not be generated.")

            grouped_docs = group_results(results)
            st.subheader(f"Top {len(grouped_docs)} Documents Found (containing {len(results)} relevant snippets):")
            
            if not grouped_docs:
                st.info("No relevant documents found for your query.")
            else:
                # Iterate through the grouped documents instead of raw chunks
                for i, group in enumerate(grouped_docs):
                    meta = group['metadata']
                    chunks = group['chunks']
                    
                    with st.container(border=True):
                        title = meta.get('title', 'No Title Found')
                        authors = meta.get('authors', 'No Authors Found')
                        year = meta.get('year', 'Unknown Year')
                        doi = meta.get('doi', '')
                        
                        source_path = meta.get('source', 'Unknown Source')
                        # Get the base filename and remove the .md extension.
                        base_name = source_path.split("\\")[-1]
                        source = base_name.split('.')[0]
                        if len(source) <= 4:
                            source = base_name[:-3]

                        st.markdown(f"### {i+1}. {title}")
                        st.markdown(f"**Authors:** {authors}")
                        st.markdown(f"**Year:** {year}")
                        
                        if doi:
                            st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
                        
                        st.caption(f"Source: {source} | Found {len(chunks)} relevant snippet(s)")

                        with st.popover("🚩 Report Document Issue", help="Flag this entire document (e.g., wrong metadata, garbled text) for review."):
                            with st.form(key=f"report_doc_{i}"):
                                st.write(f"Reporting document: **{title[:40]}...**")
                                reason = st.text_area("Issue Description:", placeholder="e.g., Wrong year, garbled text throughout...")
                                
                                if st.form_submit_button("Submit Report"):
                                    if not reason:
                                        st.warning("Please enter a reason.")
                                    else:
                                        with st.spinner("Submitting report..."):
                                            # We use the first chunk's content as a representative sample for the report log
                                            first_chunk_content = chunks[0].page_content if chunks else "No content available"
                                            if submit_report_to_sheets(meta, first_chunk_content, reason):
                                                st.success("Document reported successfully!")

                        for j, chunk in enumerate(chunks):
                            headers = [chunk.metadata.get(f'Header {h}') for h in range(1, 4) if chunk.metadata.get(f'Header {h}')]
                            header_label = " > ".join(headers) if headers else "Relevant Snippet"
                            
                            with st.expander(f"📄 Snippet {j+1}: {header_label}"):
                                st.write(chunk.page_content)

    with tab2:
        st.subheader("Find Top Authors by Subject")
        st.info("This tool searches the **Full Database** to find authors who have published most frequently on a given subject.")
        
        with st.form("author_search_form"):
            author_query = st.text_input(
                "Search Subject:", 
                placeholder="e.g., carbon capture"
            )
            # doc_scan_k = st.slider(
            #     "Number of documents to scan:", 
            #     min_value=10, max_value=200, value=50,
            #     help="How many of the most relevant documents to scan to find authors. A higher number is more thorough but slower."
            # )
            author_search_submitted = st.form_submit_button("Find Authors", type="primary", width='stretch')

        if author_search_submitted and author_query:
            # Define a fixed, large number of documents to scan
            DOC_SCAN_K = 1000
            with st.spinner(f"Searching Full Database for authors on '{author_query}' (scanning top {DOC_SCAN_K} docs)..."):
                try:
                    # Ensure we are using the full store for this feature
                    # We need the embeddings and API key which are loaded in main()
                    full_author_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
                    
                    # Perform the search to get relevant documents
                    search_results = full_author_store.similarity_search(
                        author_query, 
                        k=DOC_SCAN_K
                    )
                    
                    if not search_results:
                        st.warning("No documents found for this subject.")
                    else:
                        # Use collections.Counter for easy counting
                        author_counts = collections.Counter()
                        
                        # We need to group results first to avoid over-counting authors
                        # if multiple chunks from the same paper are returned.
                        grouped_docs = group_results(search_results)
                        
                        for group in grouped_docs:
                            meta = group['metadata']
                            authors_string = meta.get('authors', 'No Authors Found')
                            
                            if authors_string != 'No Authors Found' and authors_string is not None:
                                # Split authors string, strip whitespace, and filter out unwanted values
                                individual_authors = [
                                    name.strip() for name in authors_string.split(',')
                                    if name.strip() and name.strip().lower() not in ['not available', 'n/a']
                                ]
                                # Update counts with the cleaned list
                                author_counts.update(individual_authors)
                                
                        if not author_counts:
                            st.info("Documents were found, but no author information was attached to them.")
                        else:
                            # Get the top 10 most common authors
                            top_10_authors = author_counts.most_common(10)
                            
                            st.subheader(f"Top 10 Authors on '{author_query}'")
                            st.write(f"(From {len(grouped_docs)} unique documents found in the top {DOC_SCAN_K} relevant docs)")
                            
                            # Create DataFrame
                            df = pd.DataFrame(top_10_authors, columns=["Author", "Relevant Publications Found"])
                            
                            # 1. Calculate Y-axis limit
                            max_val = df["Relevant Publications Found"].max()
                            # Calculate nearest 5 above max (e.g., 12->15, 10->15, 9->10, 0->5)
                            ylim_top = (max_val // 5 + 1) * 5

                            # 2. Create Altair chart
                            chart = alt.Chart(df).mark_bar().encode(
                                # 3. Set X-axis: Use 'Author' column, disable axis sorting (sort=None)
                                #    AND explicitly set labelAngle to 0 to prevent rotation.
                                x=alt.X('Author', sort=None, axis=alt.Axis(labelAngle=0)),
                                
                                # 1. Set Y-axis: Use 'Relevant Publications Found'
                                #    Set scale domain from 0 to our calculated top limit
                                y=alt.Y('Relevant Publications Found', scale=alt.Scale(domain=[0, ylim_top])),
                                
                                # Add tooltips for interactivity
                                tooltip=['Author', 'Relevant Publications Found']
                            ).interactive()

                            # 3. Display the chart, using container width
                            st.altair_chart(chart, width='stretch')

                            # Old line:
                            # st.bar_chart(df.set_index("Author"), width='stretch')

                except Exception as e:
                    st.error(f"An error occurred during the author search: {e}")

    with tab3:
        st.subheader("Query Relevance Analyzer")
        st.info("Paste in a consultation or text, and this tool will extract the questions and check if the **Full Database** has relevant answers.")
        
        if not api_key_present:
            st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. This feature requires an API key to extract questions.", icon="⚠️")
        
        with st.form("query_analyzer_form"):
            consultation_text = st.text_area(
                "Paste Consultation Text Here:",
                placeholder="Paste your full consultation text... e.g., 'Question 1: Do you agree...?'",
                height=300
            )
            relevance_threshold = st.slider(
                "Relevance Threshold (Reranker Score):",
                min_value=-5.0, max_value=5.0, value=1.0, step=0.5,
                help="The minimum score from the reranker model to be considered 'Relevant'. A good starting point is 1.0."
            )
            analyze_submitted = st.form_submit_button("Analyze Questions", type="primary", width='stretch', disabled=not api_key_present)
        
        if analyze_submitted and consultation_text:
            extracted_questions = None
            extraction_token_info = None # Initialize token info
            
            with st.spinner("Step 1/2: Extracting questions from text..."):
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                # Capture both return values
                extracted_questions, extraction_token_info = extract_questions_with_llm(consultation_text)

            # Display token usage if it was returned
            if extraction_token_info:
                display_token_usage(extraction_token_info, "gpt-4o-mini", "Question Extraction")

            if extracted_questions:
                st.write(f"Found {len(extracted_questions)} questions. Now analyzing relevance against the **Full Database**...")
                
                with st.spinner(f"Step 2/2: Analyzing {len(extracted_questions)} questions... This may take a moment."):
                    try:
                        # Load the necessary models
                        reranker_model = load_reranker_model()
                        full_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
                        
                        analysis_results = []
                        progress_bar = st.progress(0, text="Analyzing...")

                        for i, question in enumerate(extracted_questions):
                            # 1. Perform initial search
                            initial_results = full_store.similarity_search(question, k=10)
                            
                            top_score = -10.0 # Default to a very low score
                            
                            if initial_results:
                                # 2. Rerank the top results
                                reranker_pairs = [(question, d.page_content) for d in initial_results]
                                reranker_scores = reranker_model.predict(reranker_pairs)
                                
                                # 3. Get the single best score
                                top_score = max(reranker_scores)
                            
                            # 4. Check relevance
                            is_relevant = top_score > relevance_threshold
                            analysis_results.append({
                                "question": question,
                                "top_score": top_score,
                                "is_relevant": is_relevant
                            })
                            progress_bar.progress((i + 1) / len(extracted_questions), text=f"Analyzing question {i+1}/{len(extracted_questions)}")
                        
                        progress_bar.empty()
                        st.subheader("Analysis Complete")
                        
                        # Display results
                        for result in analysis_results:
                            with st.container(border=True):
                                st.markdown(f"**Question:** {result['question']}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(label="Top Relevance Score", value=f"{result['top_score']:.2f}")
                                with col2:
                                    if result['is_relevant']:
                                        st.success("✅ Relevant")
                                    else:
                                        st.error("❌ Not Relevant")

                    except Exception as e:
                        st.error(f"An error occurred during relevance analysis: {e}")

            elif extracted_questions is None:
                # Error messages are already handled inside the extract_questions_with_llm function
                # (e.g., API error, JSON decode error)
                pass
            else:
                st.info("No questions were found in the provided text.")
    st.markdown("---")
    st.caption("🔒 Anonymous usage statistics are collected to help improve this tool. No personal data or IP addresses are stored.")

if __name__ == "__main__":
    main()