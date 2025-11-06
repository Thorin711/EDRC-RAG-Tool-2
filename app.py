# -*- coding: utf-8 -*-
"""
This module implements a Streamlit web application for searching research papers.
It includes RAG features, AI summarization, and a user reporting system via Google Sheets.
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

# --- NEW IMPORTS FOR GOOGLE SHEETS ---
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- QDRANT CONFIG ---
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io" 

COLLECTION_FULL = "full_papers" 
COLLECTION_JOURNAL = "journal_papers" 
COLLECTION_EDRC = "edrc_papers"

# --- GOOGLE SHEETS CONFIG ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
REPORT_SHEET_NAME = "RAG Data Reports" 

# Pricing per million tokens (Input, Output)
MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("BAAI/bge-reranker-large")

def rerank_results(query, docs, reranker_model, top_k=10):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker_model.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in reranked[:top_k]]
    return reranked_docs

# --- GOOGLE SHEETS REPORTING FUNCTION ---
def submit_report_to_sheets(doc_metadata, chunk_content, reason):
    """Appends a reporting row to Google Sheets using Streamlit secrets."""
    try:
        # Check if secrets exist before attempting connection
        if "gcp_service_account" not in st.secrets:
            st.error("Google Cloud credentials not found in secrets.")
            return False

        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], SCOPES)
        client = gspread.authorize(creds)
        # Open the spreadsheet by name. Ensure this exact name exists in your Drive.
        sheet = client.open(REPORT_SHEET_NAME).sheet1
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare the row. We truncate the snippet to avoid huge cells.
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
        st.error(f"Spreadsheet '{REPORT_SHEET_NAME}' not found. Please check the name and share it with the service account email.")
        return False
    except Exception as e:
        st.error(f"Failed to submit report: {e}")
        return False

# --- VECTOR STORE LOADERS ---
@st.cache_resource
def load_full_store(_embeddings, _url, _api_key):
    return Qdrant.from_existing_collection(
        embedding=_embeddings, collection_name=COLLECTION_FULL, url=_url, api_key=_api_key,
        content_payload_key="page_content", metadata_payload_key="metadata"
    )

@st.cache_resource
def load_journal_store(_embeddings, _url, _api_key):
    return Qdrant.from_existing_collection(
        embedding=_embeddings, collection_name=COLLECTION_JOURNAL, url=_url, api_key=_api_key,
        content_payload_key="page_content", metadata_payload_key="metadata"
    )

@st.cache_resource
def load_edrc_store(_embeddings, _url, _api_key):
    return Qdrant.from_existing_collection(
        embedding=_embeddings, collection_name=COLLECTION_EDRC, url=_url, api_key=_api_key,
        content_payload_key="page_content", metadata_payload_key="metadata"
    )

@st.cache_data
def count_tokens(text: str, model: str = "gpt-5-nano") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback for newer/unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def display_token_usage(token_info, model_name, title):
    input_tokens = token_info.get('input_tokens', 0)
    output_tokens = token_info.get('output_tokens', 0)
    total_tokens = token_info.get('total_tokens', 0)

    model_pricing = MODEL_COSTS.get(model_name, {"input": 0, "output": 0})
    cost = (input_tokens * model_pricing.get("input", 0) / 1_000_000) + \
           (output_tokens * model_pricing.get("output", 0) / 1_000_000)
    co2 = output_tokens * 0.159/50 

    with st.expander(f"Token Usage Details: {title}"):
        st.markdown(f"- **Input Tokens:** `{input_tokens}`")
        st.markdown(f"- **Output Tokens:** `{output_tokens}`")
        st.markdown(f"- **Total Tokens:** `{total_tokens}`")
        st.markdown(f"- **Estimated Cost:** `${cost:.6f}` (Model: `{model_name}`)")
        st.markdown(f"- **Estimated CO2:** `{co2:.2f} g`")

def improve_query_with_llm(user_query):
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
            temperature=0.2, max_tokens=100
        )
        return response.choices[0].message.content.strip().strip('"'), response.usage.__dict__
    except Exception as e:
        st.warning(f"Query improvement failed: {e}. Using original query.")
        return user_query, None

def summarize_results_with_llm(user_query, _search_results, model="gpt-5-nano", max_completion_tokens=10000):
    try:
        context_snippets = []
        for i, doc in enumerate(_search_results):
            title = doc.metadata.get('title', 'No Title Found')
            context_snippets.append(f"--- Source [{i+1}] ---\nTitle: {title}\nSnippet: {doc.page_content}\n---")

        full_context = "\n\n".join(context_snippets)
        prompt = f"""
        You are a research assistant. Synthesize the provided document snippets to answer the user's question. 
        Strictly base your response ONLY on the provided snippets. Cite every claim using [Number], e.g., [1], [1, 3].
        If insufficient information, state that clearly. Use bullet points. Add a ## References section at the end listing the sources.

        User's question: "{user_query}"

        --- Document Snippets ---
        {full_context}
        """
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that provides citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=1, max_completion_tokens=max_completion_tokens
        )
        return response.choices[0].message.content.strip(), response.usage.__dict__
    except Exception as e:
        st.error(f"Summary generation failed: {e}")
        return None, None

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Research Paper Search", page_icon="📚", layout="wide")

    # --- Session State Initialization ---
    if 'final_query' not in st.session_state: st.session_state.final_query = ""
    if 'search_results' not in st.session_state: st.session_state.search_results = None
    if 'original_query' not in st.session_state: st.session_state.original_query = ""
    if 'summary_generated' not in st.session_state: st.session_state.summary_generated = False
    if 'summary_content' not in st.session_state: st.session_state.summary_content = None
    if 'summary_token_info' not in st.session_state: st.session_state.summary_token_info = None
    if 'selected_collection' not in st.session_state: st.session_state.selected_collection = COLLECTION_FULL

    # Form Defaults
    if "user_query_input" not in st.session_state: st.session_state.user_query_input = ""
    if "k_results_input" not in st.session_state: st.session_state.k_results_input = 10
    if "enhanced_search_toggle" not in st.session_state: st.session_state.enhanced_search_toggle = True
    if "summary_toggle" not in st.session_state: st.session_state.summary_toggle = True
    if "start_date_input" not in st.session_state: st.session_state.start_date_input = 2015
    if "end_date_input" not in st.session_state: st.session_state.end_date_input = 2024
    if "date_filter_toggle" not in st.session_state: st.session_state.date_filter_toggle = False

    st.title("📚 Research Paper Search")
    st.write("Ask a question about your documents, and the app will find the most relevant information.")
    
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    api_key_present = bool(openai_api_key)

    if not openai_api_key:
        st.warning("`OPENAI_API_KEY` missing. AI features disabled.", icon="⚠️")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` missing. Cannot connect to database.", icon="🚨")
        st.stop()    
        
    DB_OPTIONS = {
        "Full Database": COLLECTION_FULL,
        "Journal Articles Only": COLLECTION_JOURNAL,
        "EDRC Only": COLLECTION_EDRC,
    }

    try:
        current_collection_index = list(DB_OPTIONS.values()).index(st.session_state.selected_collection)
    except ValueError:
        current_collection_index = 0

    db_choice = st.radio(
        "Select database to search:",
        options=DB_OPTIONS.keys(),
        horizontal=True,
        index=current_collection_index
    )
    selected_collection_name = DB_OPTIONS[db_choice]

    if selected_collection_name != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection_name
        # Reset all state on DB change
        st.session_state.search_results = None
        st.session_state.final_query = ""
        st.session_state.original_query = ""
        st.session_state.summary_generated = False
        st.session_state.summary_content = None
        st.session_state.summary_token_info = None
        st.rerun()

    available_models = ["gpt-5-nano", "gpt-4o-mini", "gpt-5-mini"]
    selected_model = st.selectbox("Select AI Model for Summary:", options=available_models, index=0)

    try:
        embeddings = load_embedding_model()
        if selected_collection_name == COLLECTION_FULL:
            vector_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
        elif selected_collection_name == COLLECTION_JOURNAL:
            vector_store = load_journal_store(embeddings, QDRANT_URL, qdrant_api_key)
        else:
            vector_store = load_edrc_store(embeddings, QDRANT_URL, qdrant_api_key)
            
        count_result = vector_store.client.count(collection_name=selected_collection_name, exact=True)
        st.caption(f"ℹ️ `{db_choice}` loaded with {count_result.count} documents.")
    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.stop()

    with st.form("search_form"):
        user_query = st.text_input("Ask a question:", key="user_query_input", placeholder="e.g., effects of policy on renewable energy?")
        col1, col2, col3 = st.columns([5, 2, 3])
        with col1: k_results = st.slider("Results to return:", 1, 30, key="k_results_input")
        with col2: use_enhanced_search = st.toggle("AI-Enhanced Search", disabled=not api_key_present, key="enhanced_search_toggle")
        with col3: generate_summary = st.toggle("Generate AI Summary", disabled=not api_key_present, key="summary_toggle")

        d1, d2, d3 = st.columns([2, 2, 6])
        with d1: start_date = st.number_input("Start Year", 1900, 2100, key="start_date_input")
        with d2: end_date = st.number_input("End Year", 1900, 2100, key="end_date_input")
        with d3: use_date_filter = st.checkbox("Filter by year", key="date_filter_toggle")

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted and user_query:
        # Reset results state on new search
        st.session_state.search_results = None
        st.session_state.summary_generated = False
        st.session_state.summary_content = None
        st.session_state.original_query = user_query
        st.session_state.final_query = user_query # Default fallback

        if use_enhanced_search and api_key_present:
            with st.spinner("Improving query..."):
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                improved, token_info = improve_query_with_llm(user_query)
                if token_info:
                    st.session_state.final_query = improved
                    display_token_usage(token_info, "gpt-4o-mini", "Query Enhancement")

    run_final_search = False
    if st.session_state.final_query and not st.session_state.search_results:
        with st.form("final_search_form"):
            st.info("Review suggested query before searching.")
            edited_query = st.text_area("Query:", value=st.session_state.final_query, height=100)
            use_reranker = st.toggle("Use Reranker (BGE-Large)", help="Slower but more precise ranking.")
            run_final_search = st.form_submit_button("Run Search", type="primary", use_container_width=True)

        if run_final_search:
            with st.spinner(f"Searching `{db_choice}`..."):
                try:
                    search_kwargs = {"k": k_results}
                    if use_date_filter:
                        search_kwargs["filter"] = Filter(must=[FieldCondition(key="metadata.year", range=Range(gte=start_date, lte=end_date))])

                    initial_results = vector_store.similarity_search(edited_query, **search_kwargs)
                    
                    if use_reranker:
                        with st.spinner("Re-ranking..."):
                            reranker = load_reranker_model()
                            st.session_state.search_results = rerank_results(edited_query, initial_results, reranker, top_k=k_results)
                    else:
                        st.session_state.search_results = initial_results
                except Exception as e:
                    st.error(f"Search failed: {e}")
            st.rerun()

    if st.session_state.search_results is not None:
        results = st.session_state.search_results

        # Summary Logic
        if st.session_state.summary_generated:
            with st.expander("✨ **AI-Generated Summary**", expanded=True):
                st.markdown(st.session_state.summary_content)
            if st.session_state.summary_token_info:
                 display_token_usage(st.session_state.summary_token_info, selected_model, "AI Summary")
        elif generate_summary and results and api_key_present:
            with st.form("summary_confirm"):
                st.subheader("Generate AI Summary")
                # Cost estimation logic (simplified for brevity)
                est_tokens = count_tokens(st.session_state.original_query + "".join([d.page_content for d in results]), model=selected_model)
                st.write(f"Estimated Input Tokens: ~{est_tokens}")
                if st.form_submit_button("Generate Summary", type="primary"):
                     with st.spinner("Synthesizing..."):
                         openai.api_key = st.secrets["OPENAI_API_KEY"]
                         summary, t_info = summarize_results_with_llm(st.session_state.original_query, results, model=selected_model)
                         if summary:
                             st.session_state.summary_content = summary
                             st.session_state.summary_token_info = t_info
                             st.session_state.summary_generated = True
                             st.rerun()

        # Results Display
        st.subheader(f"Top {len(results)} Results:")
        if not results:
            st.info("No relevant documents found.")
        else:
            for i, doc in enumerate(results):
                with st.container(border=True):
                    title = doc.metadata.get('title', 'No Title')
                    source = doc.metadata.get('source', 'Unknown')
                    year = doc.metadata.get('year', 'Unknown')
                    doi = doc.metadata.get('doi', '')

                    st.markdown(f"### {i+1}. {title}")
                    st.markdown(f"**Year:** {year} | **Source:** `{source}`")
                    if doi: st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")

                    with st.expander("Show content snippet"):
                        st.write(doc.page_content)

                    # --- NEW REPORTING UI ---
                    # Using st.popover for a compact "Report" button
                    with st.popover("🚩 Report Issue", help="Flag incorrect metadata or garbled text for admin review."):
                        with st.form(key=f"report_form_{i}"):
                            st.write(f"Reporting: **{title[:40]}...**")
                            reason = st.text_area("Issue Description:", placeholder="e.g., Wrong year, garbled text...")
                            if st.form_submit_button("Submit Report"):
                                if reason:
                                    with st.spinner("Submitting..."):
                                        if submit_report_to_sheets(doc.metadata, doc.page_content, reason):
                                            st.success("Report submitted! Thanks for helping improve the database.")
                                else:
                                    st.warning("Please enter a reason.")

if __name__ == "__main__":
    main()