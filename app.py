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
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Range
from langchain_huggingface import HuggingFaceEmbeddings
import openai
import tiktoken
from sentence_transformers import CrossEncoder
import collections  # Added for counting authors
import pandas as pd  # Added for displaying author results
import altair as alt # Added for custom bar chart
import json         # Added for parsing LLM-generated question lists
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- UNIVERSAL SECRET GETTER ---
def get_secret(key):
    """
    Retrieves a secret from environment variables (Hugging Face) 
    or Streamlit secrets (Local/Streamlit Cloud).
    """
    # 1. Try Environment Variable (Hugging Face / Docker)
    if key in os.environ:
        return os.environ[key]
    
    # 2. Try Streamlit Secrets (Local .toml)
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return None
# -------------------------------

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
    """Loads and caches the sentence embedding model from Hugging Face."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_full_store(_embeddings, _url, _api_key):
    """Loads and caches the FULL vector store with explicit client creation."""
    client = QdrantClient(
        url=_url, 
        api_key=_api_key,
        prefer_grpc=False 
    )
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
        prefer_grpc=False
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
        prefer_grpc=False
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
    """Counts the number of tokens in a text string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
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
    """Submits a data quality report to a Google Sheet."""
    try:
        # Note: Handling Google Service Account JSON in env vars requires parsing.
        # For now, we assume st.secrets works or the user has handled this separately.
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud credentials not found in Streamlit secrets.")
            return False

        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], SCOPES)
        client = gspread.authorize(creds)
        
        sheet = client.open(REPORT_SHEET_NAME).sheet1
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
    """Logs anonymous usage stats to the 'Usage Logs' sheet."""
    try:
        if "gcp_service_account" not in st.secrets:
            return 

        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], SCOPES)
        client = gspread.authorize(creds)
        
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
        print(f"Logging failed: {e}")

def improve_query_with_llm(user_query):
    """Improves a user's query using an LLM for better search results."""
    try:
        prompt = f"""
        You are an expert at query optimization for vector databases. Your task is to rephrase the following user's question to be more effective for a semantic search.
        Focus on using keywords and concepts found in academic research papers. Do not answer the question. Only provide the single, rephrased query.
        
        User's question: "{user_query}"
        """

        # Fix: Fetch key using get_secret
        openai.api_key = get_secret("OPENAI_API_KEY")

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
    """Uses an LLM to extract a list of questions from a block of text."""
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
        
        # Fix: Fetch key using get_secret
        openai.api_key = get_secret("OPENAI_API_KEY")

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a text-processing bot that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"} 
        )
        
        response_text = response.choices[0].message.content.strip()
        
        usage = response.usage
        token_info = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                return data, token_info
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        return value, token_info 
            
            st.error(f"LLM returned valid JSON, but not in the expected format: {response_text}")
            return None, token_info 

        except json.JSONDecodeError:
            st.error(f"Failed to decode JSON from LLM response: {response_text}")
            return None, token_info 
        
    except Exception as e:
        st.warning(f"Could not extract questions due to an API error: {e}.")
        return None, None


def summarize_results_with_llm(user_query, _search_results, model="gpt-5-nano", max_completion_tokens=10000):
    """Generates an AI-powered summary of search results with citations."""
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
        
        # Fix: Fetch key using get_secret (ensure it's set before call)
        openai.api_key = get_secret("OPENAI_API_KEY")

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
    """Groups document chunks by their source document."""
    grouped = {}
    ordered_groups = []
    for doc in results:
        key = doc.metadata.get('source') or doc.metadata.get('title') or "unknown_doc"
        
        if key not in grouped:
            group_data = {
                "metadata": doc.metadata, 
                "chunks": []
            }
            grouped[key] = group_data
            ordered_groups.append(group_data)
        
        grouped[key]["chunks"].append(doc)
    
    return ordered_groups

def main():
    """Defines the main function to run the Streamlit application."""
    st.set_page_config(page_title="Research Paper Search", page_icon="📚", layout="wide")
    
    # # --- DEBUG: LIST ALL KEYS ---
    # st.write("### Debug: Environment Keys Available")
    # # Print keys from os.environ
    # env_keys = [k for k in os.environ.keys()]
    # st.write("OS Environ Keys:", env_keys)
    
    # # Check specifically for the key
    # if "OPENAI_API_KEY" in os.environ:
        # st.success(f"✅ OPENAI_API_KEY found in os.environ! Length: {len(os.environ['OPENAI_API_KEY'])}")
    # else:
        # st.error("❌ OPENAI_API_KEY NOT found in os.environ")
    # # ----------------------------
    
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
        st.session_state.selected_collection = COLLECTION_FULL 

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
    
    # --- UNIVERSAL KEY LOADING ---
    openai_api_key = get_secret("OPENAI_API_KEY")
    qdrant_api_key = get_secret("QDRANT_API_KEY")
    
    # Try to get URL from secret, otherwise use the hardcoded default
    qdrant_url = get_secret("QDRANT_URL")
    
    api_key_present = bool(openai_api_key)

    if not openai_api_key:
        st.warning("`OPENAI_API_KEY` not found in secrets (Streamlit or Env). AI-powered features will be disabled.", icon="⚠️")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in secrets (Streamlit or Env). App cannot connect to database.", icon="🚨")
        st.stop()    
        
    DB_OPTIONS = {
        "Full Database": COLLECTION_FULL,
        "Journal Articles Only": COLLECTION_JOURNAL,
        "EDRC Only": COLLECTION_EDRC,
    }
    
    # --- Load Models and Vector Store ---
    try:
        embeddings = load_embedding_model()
        
        try:
            current_collection_index = list(DB_OPTIONS.values()).index(st.session_state.selected_collection)
        except ValueError:
            current_collection_index = 0 

        db_choice_label = list(DB_OPTIONS.keys())[current_collection_index]
        selected_collection_name = st.session_state.selected_collection
        
        # Use the securely fetched `qdrant_url` and `qdrant_api_key` here
        if selected_collection_name == COLLECTION_FULL:
            vector_store = load_full_store(embeddings, qdrant_url, qdrant_api_key)
        elif selected_collection_name == COLLECTION_JOURNAL:
            vector_store = load_journal_store(embeddings, qdrant_url, qdrant_api_key)
        else:
            vector_store = load_edrc_store(embeddings, qdrant_url, qdrant_api_key)
        
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
            index=current_collection_index,
            key="db_selection_radio"
        )
        
        selected_collection_name_from_radio = DB_OPTIONS[db_choice]

        if selected_collection_name_from_radio != st.session_state.selected_collection:
            st.session_state.selected_collection = selected_collection_name_from_radio
            
            # Clear state
            st.session_state.search_results = None
            st.session_state.final_query = ""
            st.session_state.original_query = ""
            st.session_state.summary_generated = False
            st.session_state.summary_content = None
            st.session_state.summary_token_info = None
            
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
                    key="k_results_input",
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

            date_col1, date_col2, date_col3 = st.columns([2, 2, 6])
            with date_col1:
                start_date = st.number_input(
                    "Start Year", min_value=2015, max_value=2050, step=1, value=2015,
                )
            with date_col2:
                end_date = st.number_input(
                    "End Year", min_value=2015, max_value=2050, step=1, value=2015,
                    key="end_date_input"
                )
            with date_col3:
                use_date_filter = st.checkbox(
                    "Filter by year", 
                    key="date_filter_toggle"
                )

            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

        if submitted and user_query:
            st.session_state.search_results = None
            st.session_state.final_query = ""
            st.session_state.original_query = user_query
            st.session_state.summary_generated = False
            st.session_state.summary_content = None
            st.session_state.summary_token_info = None

            if use_enhanced_search and api_key_present:
                with st.spinner("Improving query..."):
                    # Use get_secret implicitly inside the function or set it here
                    openai.api_key = get_secret("OPENAI_API_KEY") 
                    improved_query, token_info = improve_query_with_llm(user_query)

                    if token_info:
                        display_token_usage(token_info, "gpt-4o-mini", "Query Enhancement")
                        st.session_state.final_query = improved_query if improved_query else user_query
                    elif not st.session_state.final_query:
                        st.session_state.final_query = user_query
        run_final_search = False 

        if st.session_state.final_query and not st.session_state.search_results:
            with st.form("final_search_form"):
                st.info("Review and edit the query below, then click 'Run Search'.")
                edited_query = st.text_area("Suggested Query:", value=st.session_state.final_query, height=100)
                
                # use_reranker = st.toggle(
                    # "Use Reranker (BGE-Large)",
                    # help="Re-rank top search results using a cross-encoder model for higher precision.",
                    # key="reranker_toggle"
                # )
                
                run_final_search = st.form_submit_button("Run Search", type="primary", use_container_width=True)

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

            elif generate_summary and results and api_key_present:
                with st.form("summary_confirmation_form"):
                    st.subheader("Generate AI Summary")
                    st.info("A summary will be generated based on the search results. Review the estimated cost below.")

                    context_text = "\n\n".join([f"--- Source [{i+1}] ---\nTitle: {doc.metadata.get('title', 'No Title Found')}\nSnippet: {doc.page_content}\n---" for i, doc in enumerate(st.session_state.search_results)])
                    est_input_tokens = count_tokens(st.session_state.original_query + context_text, model=selected_model)
                    dynamic_max_tokens = est_input_tokens + 1000

                    model_pricing = MODEL_COSTS.get(selected_model, {"input": 0, "output": 0})
                    input_cost = (est_input_tokens * model_pricing.get("input", 0)) / 1_000_000
                    output_cost = (dynamic_max_tokens * model_pricing.get("output", 0)) / 1_000_000
                    estimated_cost = input_cost + output_cost
                    est_co2 = dynamic_max_tokens * 0.159/50 


                    st.markdown(f"- **Estimated Input Tokens:** `{est_input_tokens}`")
                    st.markdown(f"- **Max Output Tokens:** `{dynamic_max_tokens}`")
                    st.markdown(f"- **Estimated Maximum Cost:** `${estimated_cost:.4f}`")
                    st.markdown(f"- **Estimated Maximum CO2:** `{est_co2:.2f}` g")

                    proceed_with_summary = st.form_submit_button("Generate Summary", type="primary")

                if proceed_with_summary:
                    with st.spinner("Thinking..."):
                        openai.api_key = get_secret("OPENAI_API_KEY") 
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
                for i, group in enumerate(grouped_docs):
                    meta = group['metadata']
                    chunks = group['chunks']
                    
                    with st.container(border=True):
                        title = meta.get('title', 'No Title Found')
                        authors = meta.get('authors', 'No Authors Found')
                        year = meta.get('year', 'Unknown Year')
                        doi = meta.get('doi', '')
                        
                        source_path = meta.get('source', 'Unknown Source')
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
            author_search_submitted = st.form_submit_button("Find Authors", type="primary", use_container_width=True)

        if author_search_submitted and author_query:
            DOC_SCAN_K = 1000
            with st.spinner(f"Searching Full Database for authors on '{author_query}' (scanning top {DOC_SCAN_K} docs)..."):
                try:
                    # Pass the secure URL/KEY here too
                    full_author_store = load_full_store(embeddings, qdrant_url, qdrant_api_key)
                    
                    search_results = full_author_store.similarity_search(
                        author_query, 
                        k=DOC_SCAN_K
                    )
                    
                    if not search_results:
                        st.warning("No documents found for this subject.")
                    else:
                        author_counts = collections.Counter()
                        grouped_docs = group_results(search_results)
                        
                        for group in grouped_docs:
                            meta = group['metadata']
                            authors_string = meta.get('authors', 'No Authors Found')
                            
                            if authors_string != 'No Authors Found' and authors_string is not None:
                                individual_authors = [
                                    name.strip() for name in authors_string.split(',')
                                    if name.strip() and name.strip().lower() not in ['not available', 'n/a']
                                ]
                                author_counts.update(individual_authors)
                                
                        if not author_counts:
                            st.info("Documents were found, but no author information was attached to them.")
                        else:
                            top_10_authors = author_counts.most_common(10)
                            
                            st.subheader(f"Top 10 Authors on '{author_query}'")
                            st.write(f"(From {len(grouped_docs)} unique documents found in the top {DOC_SCAN_K} relevant docs)")
                            
                            df = pd.DataFrame(top_10_authors, columns=["Author", "Relevant Publications Found"])
                            
                            max_val = df["Relevant Publications Found"].max()
                            ylim_top = (max_val // 5 + 1) * 5

                            chart = alt.Chart(df).mark_bar().encode(
                                x=alt.X('Author', sort=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('Relevant Publications Found', scale=alt.Scale(domain=[0, ylim_top])),
                                tooltip=['Author', 'Relevant Publications Found']
                            ).interactive()

                            st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during the author search: {e}")

    with tab3:
        st.subheader("Query Relevance Analyzer")
        st.info("Paste in a consultation or text, and this tool will extract the questions and check if the **Full Database** has relevant answers.")
        
        if not api_key_present:
            st.warning("`OPENAI_API_KEY` not found in secrets. This feature requires an API key to extract questions.", icon="⚠️")
        
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
            analyze_submitted = st.form_submit_button("Analyze Questions", type="primary", use_container_width=True, disabled=not api_key_present)
        
        if analyze_submitted and consultation_text:
            extracted_questions = None
            extraction_token_info = None 
            
            with st.spinner("Step 1/2: Extracting questions from text..."):
                openai.api_key = get_secret("OPENAI_API_KEY") 
                extracted_questions, extraction_token_info = extract_questions_with_llm(consultation_text)

            if extraction_token_info:
                display_token_usage(extraction_token_info, "gpt-4o-mini", "Question Extraction")

            if extracted_questions:
                st.write(f"Found {len(extracted_questions)} questions. Now analyzing relevance against the **Full Database**...")
                
                with st.spinner(f"Step 2/2: Analyzing {len(extracted_questions)} questions... This may take a moment."):
                    try:
                        reranker_model = load_reranker_model()
                        # Use secure keys
                        full_store = load_full_store(embeddings, qdrant_url, qdrant_api_key)
                        
                        analysis_results = []
                        progress_bar = st.progress(0, text="Analyzing...")

                        for i, question in enumerate(extracted_questions):
                            initial_results = full_store.similarity_search(question, k=10)
                            
                            top_score = -10.0 
                            
                            if initial_results:
                                reranker_pairs = [(question, d.page_content) for d in initial_results]
                                reranker_scores = reranker_model.predict(reranker_pairs)
                                top_score = max(reranker_scores)
                            
                            is_relevant = top_score > relevance_threshold
                            analysis_results.append({
                                "question": question,
                                "top_score": top_score,
                                "is_relevant": is_relevant
                            })
                            progress_bar.progress((i + 1) / len(extracted_questions), text=f"Analyzing question {i+1}/{len(extracted_questions)}")
                        
                        progress_bar.empty()
                        st.subheader("Analysis Complete")
                        
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
                pass
            else:
                st.info("No questions were found in the provided text.")
    st.markdown("---")
    st.caption("🔒 Anonymous usage statistics are collected to help improve this tool. No personal data or IP addresses are stored.")

if __name__ == "__main__":
    main()