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
from langchain_qdrant import Qdrant
from qdrant_client.http.models import FieldCondition, Range
import openai
import tiktoken

# --- CONFIGURATION ---
# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- QDRANT CONFIG ---
# URL from your Qdrant Cloud dashboard
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io" 

# --- !! IMPORTANT !! ---
# You must replace these with the *exact* names of your collections in Qdrant
COLLECTION_FULL = "full_papers" 
COLLECTION_JOURNAL = "journal_papers" 
COLLECTION_EDRC = "edrc_papers"

# Pricing per million tokens (Input, Output)
MODEL_COSTS = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},  # Placeholder cost, update as needed
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00}, # Placeholder cost, update as needed
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}


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
def load_vector_store(_embeddings, _collection_name, _url, _api_key):
    """Loads and caches a Qdrant vector store from the cloud.

    This function initializes a Qdrant vector store by connecting to
    an existing collection in Qdrant Cloud.

    Args:
        _embeddings (langchain_core.embeddings.Embeddings): The embedding
            function to use with the vector store.
        _collection_name (str): The name of the collection in Qdrant.
        _url (str): The URL of the Qdrant Cloud instance.
        _api_key (str): The API key for the Qdrant Cloud instance.

    Returns:
        langchain_qdrant.Qdrant: The loaded vector store instance.
    """
    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=_collection_name,
        url=_url,
        api_key=_api_key,
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

    # --- Initialize Session State ---
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

    st.title("📚 Research Paper Search")
    st.write("Ask a question about your documents, and the app will find the most relevant information.")
    
    # Check for API keys
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    
    api_key_present = bool(openai_api_key) # For disabling AI features

    if not openai_api_key:
        st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. AI-powered features will be disabled.", icon="⚠️")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect to database.", icon="🚨")
        st.stop()    if not api_key_present:
        st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. AI-powered features will be disabled.", icon="⚠️")

# Define database options (using Qdrant collection names from Step 3)
    DB_OPTIONS = {
        "Full Database": COLLECTION_FULL,
        "Journal Articles Only": COLLECTION_JOURNAL,
        "EDRC Only": COLLECTION_EDRC,
    }

    db_choice = st.radio(
        "Select database to search:",
        options=DB_OPTIONS.keys(),
        horizontal=True,
    )
    
    selected_collection_name = DB_OPTIONS[db_choice]

    # --- Model Selection ---
    available_models = ["gpt-5-nano", "gpt-4o-mini", "gpt-5-mini", "gpt-5"]
    selected_model = st.selectbox(
        "Select AI Model for Summary:",
        options=available_models,
        index=0,  # Default to gpt-5-nano
        help="Choose the model for AI summarization. Query enhancement is fixed to gpt-4o-mini for efficiency."
    )

    try:
        embeddings = load_embedding_model()
        vector_store = load_vector_store(
            embeddings, 
            selected_collection_name, 
            QDRANT_URL, 
            qdrant_api_key
        )
        
        # Count documents using the Qdrant client
        count_result = vector_store.client.count(
            collection_name=selected_collection_name, 
            exact=True
        )
        st.caption(f"ℹ️ `{db_choice}` loaded with {count_result.count} documents.")
        
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

        # --- Date Range Selection ---
        date_col1, date_col2, date_col3 = st.columns([2, 2, 6])
        with date_col1:
            start_date = st.number_input("Start Year", min_value=1900, max_value=2100, value=2015, step=1)
        with date_col2:
            end_date = st.number_input("End Year", min_value=1900, max_value=2100, value=2024, step=1)
        with date_col3:
            use_date_filter = st.checkbox("Filter by year", value=False)

        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

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
            with st.spinner(f"Searching `{db_choice}`..."):
                try:
                    # Build search arguments
                    search_kwargs = {"k": k_results}
                    if use_date_filter:
                        search_kwargs["filter"] = {
                            "must": [
                                FieldCondition(
                                    key="year", # Assumes your metadata field is named 'year'
                                    range=Range(gte=start_date, lte=end_date)
                                )
                            ]
                        }
                    
                    results = vector_store.similarity_search(user_query, **search_kwargs)
                    st.session_state.search_results = results
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
                    # Build search arguments
                    search_kwargs = {"k": k_results}
                    if use_date_filter:
                        search_kwargs["filter"] = {
                            "must": [
                                FieldCondition(
                                    key="year", # Assumes your metadata field is named 'year'
                                    range=Range(gte=start_date, lte=end_date)
                                )
                            ]
                        }

                    st.session_state.search_results = vector_store.similarity_search(query_to_use, **search_kwargs)
                except Exception as e:
                    st.error(f"An error occurred during the search: {e}")
            st.rerun() # Rerun to display results

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
                    title = doc.metadata.get('title', 'No Title Found')
                    authors = doc.metadata.get('authors', 'No Authors Found')
                    source_path = doc.metadata.get('source', 'Unknown Source')
                    # Get the base filename and remove the .md extension.
                    base_name = source_path.split("\\")[-1]
                    source = base_name.split('.')[0]
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
