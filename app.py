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

The application relies on Streamlit for the UI, LangChain for vector store
management, Hugging Face for embedding models, and OpenAI for the LLM-powered
features.
"""

import streamlit as st
import os
import requests
import zipfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import openai
import tiktoken

# --- CONFIGURATION ---
DB_FULL_PATH = './vector_db_full'
DB_JOURNAL_PATH = './vector_db_journals'
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DB_FULL_URL = "https://github.com/Thorin711/EDRC-RAG-Tool-2/releases/download/v0.2/vector_db_full.zip"
DB_JOURNAL_URL = "https://github.com/Thorin711/EDRC-RAG-Tool-2/releases/download/v0.2/vector_db_journals.zip"

# Pricing per million tokens (Input, Output)
MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},  # Placeholder cost, update as needed
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00}, # Placeholder cost, update as needed
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}

def download_and_unzip_db(url, dest_folder, zip_name):
    """Downloads and unzips a vector database if it's not already present.

    This function checks for the existence of a destination folder. If it
    doesn't exist, it downloads a ZIP file from the given URL, displays a
    progress bar in the Streamlit interface, unzips the file into the
    current directory, and then cleans up the downloaded ZIP file.

    Args:
        url (str): The URL of the database ZIP file.
        dest_folder (str): The path to the target directory where the database
                           should exist. This is used to check if the download
                           is necessary.
        zip_name (str): The filename for the downloaded ZIP file.
    """
    if not os.path.exists(dest_folder):
        st.info(f"Database for '{os.path.basename(dest_folder)}' not found. Downloading...")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = st.progress(0, text=f"Downloading {zip_name}...")
                chunk_size = 8192
                downloaded_size = 0

                zip_path = os.path.join(".", zip_name)  # Save zip in root
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = min(int((downloaded_size / total_size) * 100), 100)
                            progress_bar.progress(progress, text=f"Downloading {zip_name}... {progress}%")

            progress_bar.empty()
            with st.spinner(f"Unzipping {zip_name}..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('.')

            os.remove(zip_path)  # Clean up the zip file
            st.success(f"Database '{os.path.basename(dest_folder)}' set up successfully!")
            st.rerun()  # Rerun the script to load the DB
        except Exception as e:
            st.error(f"Failed to download or unzip database from {url}. Error: {e}")
            st.stop()


# --- CACHING ---
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
def load_vector_store(_embeddings, db_dir):
    """Loads and caches a persistent Chroma vector store from disk.

    This function initializes a `Chroma` vector store from a specified
    directory on disk. It uses the provided embedding function to handle
_   queries. The `@st.cache_resource` decorator caches the loaded vector
    store for efficiency across Streamlit app reruns.

    Args:
        _embeddings (langchain_core.embeddings.Embeddings): The embedding
            function to use with the vector store.
        db_dir (str): The directory path where the persistent vector store
                      is located.

    Returns:
        langchain_chroma.Chroma: The loaded vector store instance.
    """
    return Chroma(persist_directory=db_dir, embedding_function=_embeddings)


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

    with st.expander(f"Token Usage Details: {title}"):
        st.markdown(f"- **Input Tokens:** `{input_tokens}`")
        st.markdown(f"- **Output Tokens:** `{output_tokens}`")
        st.markdown(f"- **Total Tokens:** `{total_tokens}`")
        st.markdown(f"- **Estimated Cost:** `${cost:.6f}` (Model: `{model_name}`)")

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

            7. References Section: After your summary, add a ## References section. List all the provided document snippets numerically, corresponding to your in-line citations.

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

    # --- Initialize Session State ---
    if 'final_query' not in st.session_state:
        st.session_state.final_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'original_query' not in st.session_state:
        st.session_state.original_query = ""

    # --- Check for and download databases on startup ---
    download_and_unzip_db(DB_FULL_URL, DB_FULL_PATH, "vector_db_full.zip")
    download_and_unzip_db(DB_JOURNAL_URL, DB_JOURNAL_PATH, "vector_db_journals.zip")

    st.title("📚 Research Paper Search")
    st.write("Ask a question about your documents, and the app will find the most relevant information.")
    
    api_key_present = "OPENAI_API_KEY" in st.secrets
    if not api_key_present:
        st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. AI-powered features will be disabled.", icon="⚠️")

    db_choice = st.radio(
        "Select database to search:",
        ("Full Database", "Journal Articles Only"),
        horizontal=True,
    )

    # --- Model Selection ---
    available_models = ["gpt-5-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5"]
    selected_model = st.selectbox(
        "Select AI Model for Summary:",
        options=available_models,
        index=0,  # Default to gpt-5-nano
        help="Choose the model for AI summarization. Query enhancement is fixed to gpt-4o-mini for efficiency."
    )

    selected_db_path = DB_FULL_PATH if db_choice == "Full Database" else DB_JOURNAL_PATH
        
    try:
        embeddings = load_embedding_model()
        vector_store = load_vector_store(embeddings, selected_db_path)
        st.caption(f"ℹ️ `{db_choice}` loaded with {vector_store._collection.count()} documents.")
    except Exception as e:
        st.error(f"An error occurred while loading the models or database: {e}")
        st.stop()

    with st.form("search_form"):
        user_query = st.text_input(
            "Ask a question:", 
            placeholder="e.g., What are the effects of policy on renewable energy adoption?"
        )
        
        col1, col2, col3 = st.columns([5, 2, 3])
        with col1:
            k_results = st.slider("Number of results to return:", min_value=1, max_value=30, value=10)
        with col2:
            use_enhanced_search = st.toggle(
                "AI-Enhanced Search", 
                value=True, 
                help="Uses an AI model to rephrase your query.",
                disabled=not api_key_present
            )
        with col3:
            generate_summary = st.toggle(
                "Generate AI Summary",
                value=True,
                help="Uses an AI model to summarize the search results.",
                disabled=not api_key_present
            )

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    # --- Main Logic ---
    if submitted and user_query:
        # Clear previous results and reset state on new submission
        st.session_state.search_results = None
        st.session_state.final_query = ""
        st.session_state.original_query = user_query

        if use_enhanced_search and api_key_present:
            with st.spinner("Improving query..."):
                est_input_tokens = count_tokens(user_query, model="gpt-4o-mini")
                st.caption(f"Estimated input tokens for query enhancement: ~{est_input_tokens}")

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
            with st.spinner(f"Searching `{db_choice}`..."):
                try:
                    st.session_state.search_results = vector_store.similarity_search(user_query, k=k_results)
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
            st.rerun() # Rerun to display results immediately

    # --- Display Enhanced Query for Editing ---
    if st.session_state.final_query and not st.session_state.search_results:
        with st.form("final_search_form"):
            st.info("Review and edit the query below, then click 'Run Search'.")
            edited_query = st.text_area("Suggested Query:", value=st.session_state.final_query, height=100)
            run_final_search = st.form_submit_button("Run Search", type="primary", use_container_width=True)

        if run_final_search:
            query_to_use = edited_query
            with st.spinner(f"Searching `{db_choice}` with final query..."):
                try:
                    st.session_state.search_results = vector_store.similarity_search(query_to_use, k=k_results)
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
            st.rerun() # Rerun to display results

    # --- Display Search Results ---
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if generate_summary and results and api_key_present:
            with st.spinner("Thinking..."):
                # Estimate input tokens to calculate a dynamic max_completion_tokens for the output
                context_text = "\n\n".join([f"--- Source [{i+1}] ---\nTitle: {doc.metadata.get('title', 'No Title Found')}\nSnippet: {doc.page_content}\n---" for i, doc in enumerate(st.session_state.search_results)])
                # A more accurate estimation of the full prompt sent to the model
                est_input_tokens = count_tokens(st.session_state.original_query + context_text, model=selected_model)
                # Provide a reasonable estimate for the output. This is a heuristic.
                est_output_tokens = 800 
                st.caption(f"Estimated input tokens for summary: ~{est_input_tokens}")
                st.caption(f"Estimated output tokens for summary: ~{est_output_tokens}")

                dynamic_max_tokens = est_input_tokens + 1000

                openai.api_key = st.secrets["OPENAI_API_KEY"]
                summary, token_info = summarize_results_with_llm(st.session_state.original_query, st.session_state.search_results, model=selected_model, max_completion_tokens=dynamic_max_tokens)
                
                if summary and token_info:
                    with st.expander("✨ **AI-Generated Summary**", expanded=True):
                        st.markdown(summary)
                    display_token_usage(token_info, selected_model, "AI Summary")
                else:
                    st.warning("The AI summary could not be generated.")

        st.subheader(f"Top {len(results)} Relevant Documents from {db_choice}:")
        if not results:
            st.info("No relevant documents found for your query.")
        else:
            for i, doc in enumerate(results):
                with st.container(border=True):
                    title = doc.metadata.get('title', 'No Title Found')
                    authors = doc.metadata.get('authors', 'No Authors Found')
                    source = doc.metadata.get('source', 'Unknown Source')
                    source = source.split("\\")[-1].split(".md")[0]
                    year = doc.metadata.get('year', 'Unknown Year')
                    doi = doc.metadata.get('doi', '')
                    
                    st.markdown(f"### {i+1}. {title}")
                    
                    headers = [doc.metadata.get(f'Header {i}') for i in range(1, 4) if doc.metadata.get(f'Header {i}')]
                    if headers:
                        st.markdown(f"**Section:** {' > '.join(headers)}")

                    st.markdown(f"**Authors:** {authors}")
                    st.markdown(f"**Year:** {year}")
                    
                    if doi:
                        st.markdown(f"**DOI:** {doi}")
                    
                    with st.expander("Show content snippet"):
                        st.write(doc.page_content)

                    st.caption(f"Source: {source}")

if __name__ == "__main__":
    main()
