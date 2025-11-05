# -*- coding: utf-8 -*-
"""
This module implements a Streamlit web application for searching research papers.

The application serves as a retrieval-augmented generation (RAG) tool. It allows
users to query a pre-built vector database of research documents. Key features
include:
- On-demand downloading and caching of vector databases from a remote source.
- A user interface to select different databases (e.g., full corpus vs. journals).
- AI-powered query enhancement to improve search relevance.
- **(New)** Cross-encoder reranking for improved result precision.
- AI-powered summarization of search results with citations.
- A clear, interactive display of search results, including document metadata
  and content snippets.

The application relies on Streamlit for the UI, LangChain for vector store
management, Hugging Face for embedding models, and OpenAI for the LLM-powered
features.
"""

# --- NEW DEPENDENCIES ---
# You will need to install these:
# pip install langchain-community sentence-transformers
# ---

import streamlit as st
import os
from langchain_qdrant import Qdrant
from qdrant_client.http.models import Filter, FieldCondition, Range
from langchain_huggingface import HuggingFaceEmbeddings
import openai
import tiktoken
from langchain.retrievers.contextual_compression.ContextualCompressionRetrieve import ContextualCompressionRetriever
from langchain_community.document_compressors.cross_encoder import CrossEncoderReranker

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- QDRANT CONFIG ---
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io" 

# --- RERANKER CONFIG (NEW) ---
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
RERANKER_FETCH_K = 25 # Number of docs to fetch from Qdrant for reranking

COLLECTION_FULL = "full_papers" 
COLLECTION_JOURNAL = "journal_papers" 
COLLECTION_EDRC = "edrc_papers"

# Pricing per million tokens (Input, Output)
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


# --- START: MODIFIED SECTION (EXPLICIT CACHING) ---
# Replaced the single load_vector_store with three explicit functions
# to prevent caching conflicts.

@st.cache_resource
def load_full_store(_embeddings, _url, _api_key):
    """Loads and caches the FULL vector store."""
    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=COLLECTION_FULL,
        url=_url,
        api_key=_api_key,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )

@st.cache_resource
def load_journal_store(_embeddings, _url, _api_key):
    """Loads and caches the JOURNAL vector store."""
    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=COLLECTION_JOURNAL,
        url=_url,
        api_key=_api_key,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )

@st.cache_resource
def load_edrc_store(_embeddings, _url, _api_key):
    """Loads and caches the EDRC vector store."""
    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=COLLECTION_EDRC,
        url=_url,
        api_key=_api_key,
        content_payload_key="page_content", 
        metadata_payload_key="metadata"
    )
# --- END: MODIFIED SECTION ---

# --- NEW RERANKER CACHE ---
@st.cache_resource
def load_reranker():
    """Loads and caches the CrossEncoderReranker."""
    # The 'top_n' here is a default; we will override it dynamically
    # based on the user's 'k_results' slider.
    return CrossEncoderReranker(model_name=RERANKER_MODEL_NAME, top_n=10)
# --- END NEW RERANKER CACHE ---


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

def improve_query_with_llm(user_query):
    """Improves a user's query using an LLM for better search results.

    This function sends the user's query to an OpenAI model with a prompt
    that asks it to rephrase the query for a vector database search, focusing
    on academic keywords. The `@st.cache_data` decorator caches the result
    to avoid repeated API calls for the same query.

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


def summarize_results_with_llm(user_query, _search_results, model="gpt-5-nano", max_completion_tokens=10000):
    """Generates an AI-powered summary of search results with citations.

    This function constructs a detailed prompt containing the user's original
    question and snippets from the search results. It asks an OpenAI model to
    synthesize an answer based *only* on the provided context and to cite
    the sources for each piece of information. The `@st.cache_data` decorator
    caches the summary to prevent regenerating it on every app rerun.

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

            7. You must use bullet point formatting for the output.

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


def main():
    """Defines the main function to run the Streamlit application.

    This function sets up the Streamlit page configuration, handles the initial
    database checks and downloads, and builds the user interface. The UI
    includes a title, database selection, search form with options for the
    number of results and AI features, and a results display area. It manages
    the application's state and logic for handling user input and search
    execution.
    """
    st.set_page_config(page_title="Research Paper Search", page_icon="📚", layout="wide")

    # --- START: MODIFIED SECTION (State Initialization) ---
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
    if "start_date_input" not in st.session_state:
        st.session_state.start_date_input = 2015
    if "end_date_input" not in st.session_state:
        st.session_state.end_date_input = 2024
    if "date_filter_toggle" not in st.session_state:
        st.session_state.date_filter_toggle = False
    
    # --- ADDED RERANKER STATE ---
    if "reranker_toggle" not in st.session_state:
        st.session_state.reranker_toggle = True # Default to on
    # --- END: MODIFIED SECTION ---

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

    # --- START: MODIFIED SECTION (Stateful Radio Button) ---
    # Find the index of the collection currently in session state
    try:
        current_collection_index = list(DB_OPTIONS.values()).index(st.session_state.selected_collection)
    except ValueError:
        current_collection_index = 0 # Default to first item if state is invalid

    db_choice = st.radio(
        "Select database to search:",
        options=DB_OPTIONS.keys(),
        horizontal=True,
        index=current_collection_index  # Use the index from session state
    )
    
    selected_collection_name = DB_OPTIONS[db_choice]

    # If the user selected a new database, update the state and clear old results
    if selected_collection_name != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection_name
        
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
        st.session_state.reranker_toggle = True # Reset reranker toggle
        
        st.rerun() # Rerun to apply the change and show a clean state
    # --- END: MODIFIED SECTION ---

    available_models = ["gpt-5-nano", "gpt-4o-mini", "gpt-5-mini"]
    selected_model = st.selectbox(
        "Select AI Model for Summary:",
        options=available_models,
        index=0,
        help="Choose the model for AI summarization. Query enhancement is fixed to gpt-4o-mini for efficiency."
    )

    try:
        embeddings = load_embedding_model()
        reranker = load_reranker() # <-- LOAD RERANKER
        
        # --- START: MODIFIED SECTION (Explicit Cache Loading) ---
        if selected_collection_name == COLLECTION_FULL:
            vector_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
        elif selected_collection_name == COLLECTION_JOURNAL:
            vector_store = load_journal_store(embeddings, QDRANT_URL, qdrant_api_key)
        else:
            vector_store = load_edrc_store(embeddings, QDRANT_URL, qdrant_api_key)
        # --- END: MODIFIED SECTION ---
        
        count_result = vector_store.client.count(
            collection_name=selected_collection_name, 
            exact=True
        )
        st.caption(f"ℹ️ `{db_choice}` (collection: `{selected_collection_name}`) loaded with {count_result.count} documents.")
        
    except Exception as e:
        st.error(f"An error occurred while loading the models or database: {e}")
        st.stop()

    # --- START: MODIFIED SECTION (Updated Form with Reranker) ---
    with st.form("search_form"):
        user_query = st.text_input(
            "Ask a question:", 
            placeholder="e.g., What are the effects of policy on renewable energy adoption?",
            key="user_query_input" 
        )
        
        # --- MODIFIED COLUMNS FOR RERANKER ---
        col1, col2, col3, col4 = st.columns([5, 2, 3, 2])
        with col1:
            k_results = st.slider(
                "Number of results to return:", 
                min_value=1, 
                max_value=RERANKER_FETCH_K - 5, # Max results must be < fetch_k
                key="k_results_input",
                help=f"The final number of results to display. The app will fetch {RERANKER_FETCH_K} and rerank to find the best ones."
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
        # --- NEW RERANKER TOGGLE ---
        with col4:
            use_reranker = st.toggle(
                "Enable Reranker",
                help="Uses a more accurate (but slower) model to rerank results.",
                key="reranker_toggle"
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

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)
    # --- END: MODIFIED SECTION ---


    # --- Main Logic ---
    if submitted and user_query:
        # Clear previous results and reset state on new submission
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
        else:
            # If not using enhanced search, just run the search directly
            # --- START: RERANKER LOGIC BLOCK 1 ---
            spinner_msg = f"Searching and reranking `{db_choice}`..." if use_reranker else f"Searching `{db_choice}`..."
            with st.spinner(spinner_msg):
                try:
                    # 1. Build the filter if needed
                    filter_dict = None
                    if use_date_filter:
                        filter_dict = Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.year", 
                                    range=Range(gte=start_date, lte=end_date)
                                )
                            ]
                        )
                    
                    if use_reranker:
                        # 2. Configure the base retriever
                        base_retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": RERANKER_FETCH_K, # Fetch 25
                                "filter": filter_dict
                            }
                        )
                        
                        # 3. Configure the reranker with the user's desired final 'k'
                        reranker.top_n = k_results # User's slider value
                        
                        # 4. Create the compression retriever
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=reranker,
                            base_retriever=base_retriever
                        )
                        
                        # 5. Run the search
                        results = compression_retriever.invoke(user_query)
                    
                    else:
                        # Original search logic
                        search_kwargs = {"k": k_results, "filter": filter_dict}
                        results = vector_store.similarity_search(user_query, **search_kwargs)
                    
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
            st.rerun() # Rerun to display results immediately
            # --- END: RERANKER LOGIC BLOCK 1 ---

    # --- Display Enhanced Query for Editing ---
    if st.session_state.final_query and not st.session_state.search_results:
        with st.form("final_search_form"):
            st.info("Review and edit the query below, then click 'Run Search'.")
            edited_query = st.text_area("Suggested Query:", value=st.session_state.final_query, height=100)
            run_final_search = st.form_submit_button("Run Search", type="primary", use_container_width=True)

        if run_final_search:
            query_to_use = edited_query
            # --- START: RERANKER LOGIC BLOCK 2 ---
            spinner_msg = f"Searching and reranking `{db_choice}`..." if use_reranker else f"Searching `{db_choice}`..."
            with st.spinner(spinner_msg):
                try:
                    # 1. Build the filter if needed
                    filter_dict = None
                    if use_date_filter:
                        filter_dict = Filter(
                            must=[
                                FieldCondition(
                                    key="metadata.year", 
                                    range=Range(gte=start_date, lte=end_date)
                                )
                            ]
                        )
                    
                    if use_reranker:
                        # 2. Configure the base retriever
                        base_retriever = vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": RERANKER_FETCH_K, # Fetch 25
                                "filter": filter_dict
                            }
                        )
                        
                        # 3. Configure the reranker with the user's desired final 'k'
                        reranker.top_n = k_results # User's slider value
                        
                        # 4. Create the compression retriever
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=reranker,
                            base_retriever=base_retriever
                        )
                        
                        # 5. Run the search
                        results = compression_retriever.invoke(query_to_use)
                    
                    else:
                        # Original search logic
                        search_kwargs = {"k": k_results, "filter": filter_dict}
                        results = vector_store.similarity_search(query_to_use, **search_kwargs)

                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
            st.rerun() # Rerun to display results
            # --- END: RERANKER LOGIC BLOCK 2 ---

    # --- Display Search Results ---
    if st.session_state.search_results is not None:
        results = st.session_state.search_results

        # --- AI Summary Section ---
        # Display summary if it has already been generated
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

        st.subheader(f"Top {len(results)} Relevant Documents from {db_choice}:")
        if not results:
            st.info("No relevant documents found for your query.")
        else:
            for i, doc in enumerate(results):
                with st.container(border=True):
                    # --- Access metadata correctly ---
                    title = doc.metadata.get('title', 'No Title Found')
                    authors = doc.metadata.get('authors', 'No Authors Found')
                    source_path = doc.metadata.get('source', 'Unknown Source')
                    # Get the base filename and remove the .md extension.
                    base_name = source_path.split("\\")[-1]
                    source = base_name.split('.')[0]
                    if len(source) <= 4:
                        source = base_name[:-3]
                    year = doc.metadata.get('year', 'Unknown Year')
                    doi = doc.metadata.get('doi', '')
                    
                    st.markdown(f"### {i+1}. {title}")
                    
                    headers = [doc.metadata.get(f'Header {i}') for i in range(1, 4) if doc.metadata.get(f'Header {i}')]
                    if headers:
                        st.markdown(f"**Section:** {' > '.join(headers)}")

                    st.markdown(f"**Authors:** {authors}")
                    st.markdown(f"**Year:** {year}")
                    
                    if doi:
                        st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
                    # --- This is the updated code ---
                    with st.expander("Show content snippet"):
                        # 'doc.page_content' is correct because we set
                        # content_payload_key="page_content"
                        st.write(doc.page_content) 

                    st.caption(f"Source: {source}")

if __name__ == "__main__":
    main()