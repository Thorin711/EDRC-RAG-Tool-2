# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:36:42 2025

@author: td00654
"""

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import openai

# --- CONFIGURATION ---
DB_FULL_PATH = './vector_db_full'
DB_JOURNAL_PATH = './vector_db_journals'
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- CACHING ---
@st.cache_resource
def load_embedding_model():
    """Loads the embedding model from Hugging Face."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_vector_store(_embeddings, db_dir):
    """Loads the vector store from the specified persistent directory."""
    if not os.path.exists(db_dir):
        st.error(f"Database not found at {db_dir}! Please run `create_vectordb.py` script.")
        st.stop()
    return Chroma(persist_directory=db_dir, embedding_function=_embeddings)

@st.cache_data
def improve_query_with_llm(user_query):
    """
    Improves the user's query using an OpenAI model for better vector search results.
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
            temperature=0.0,
            max_tokens=100
        )
        improved_query = response.choices[0].message.content.strip()
        return improved_query.strip('"')
    except Exception as e:
        st.warning(f"Could not improve query due to an API error: {e}. Using the original query.")
        return user_query

@st.cache_data
def summarize_results_with_llm(user_query, _search_results):
    """
    Generates a summary of the search results with citations using an OpenAI model.
    """
    try:
        # Prepare the context from the search results, including titles for citations
        context_snippets = []
        for i, doc in enumerate(_search_results):
            title = doc.metadata.get('title', 'No Title Found')
            citation_marker = f"[{i+1}]"
            context_snippets.append(f"--- Source {citation_marker} ---\nTitle: {title}\nSnippet: {doc.page_content}\n---")

        full_context = "\n\n".join(context_snippets)

        prompt = f"""
        You are a research assistant. Your task is to synthesize the provided document snippets to answer the user's question. Follow these rules strictly:
        1.  Base your summary *only* on the information given in the "Document Snippets" section. Do not use any external knowledge.
        2.  After each piece of information or sentence you write, you **MUST** cite the source(s) it came from using the citation marker, e.g., [1], [2], or [1][3].
        3.  After the summary, add a "References" section and list all the sources you used with their full titles and corresponding citation marker.
        4.  If the context does not contain enough information to answer the question, state that.
        5.  Use bullet point format for the summary.

        User's question: "{user_query}"

        --- Document Snippets ---
        {full_context}

        --- Synthesized Answer ---
        [Your summary with citations goes here]

        **References**
        [Your numbered list of references goes here, e.g., "[1] Title of the first paper."]
        """
        
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that provides citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=1, 
            max_completion_tokens=5000 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Could not generate summary due to an API error: {e}")
        return None

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Research Paper Search", page_icon="📚", layout="wide")
    
    st.title("📚 Research Paper Search")
    st.write("Ask a question about your documents, and the app will find the most relevant information.")
    
    api_key_present = "OPENAI_API_KEY" in st.secrets
    if not api_key_present:
        st.warning("`OPENAI_API_KEY` not found in Streamlit secrets. AI-powered features will be disabled.", icon="⚠️")

    # --- Database selection widget ---
    db_choice = st.radio(
        "Select database to search:",
        ("Full Database", "Journal Articles Only"),
        horizontal=True,
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
            k_results = st.slider("Number of results to return:", min_value=1, max_value=10, value=5)
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

    # --- ✅ MODIFIED: Removed the st.empty() placeholder for smoother rendering ---
    if submitted and user_query:
        st.cache_data.clear()
        
        query_to_use = user_query
        
        if use_enhanced_search and api_key_present:
            with st.spinner("Improving query..."):
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                improved_query = improve_query_with_llm(user_query)
                if improved_query.lower() != user_query.lower():
                    st.info(f"Searching with improved query: **{improved_query}**")
                    query_to_use = improved_query
        
        with st.spinner(f"Searching `{db_choice}`..."):
            try:
                results = vector_store.similarity_search(query_to_use, k=k_results)
                
                if generate_summary and results and api_key_present:
                    st.info("Generating AI summary with citations...")
                    with st.spinner("Synthesizing answer..."):
                        openai.api_key = st.secrets["OPENAI_API_KEY"]
                        summary = summarize_results_with_llm(user_query, results)
                        if summary:
                            with st.expander("✨ **AI-Generated Summary**", expanded=True):
                                st.markdown(summary)
                        else:
                            st.warning("The AI summary could not be generated.")

                st.subheader(f"Top {len(results)} Relevant Documents from `{db_choice}`:")

                if not results:
                    st.info("No relevant documents found for your query.")
                else:
                    for i, doc in enumerate(results):
                        with st.container(border=True):
                            title = doc.metadata.get('title', 'No Title Found')
                            authors = doc.metadata.get('authors', 'No Authors Found')
                            source = doc.metadata.get('source', 'Unknown Source')
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
                            
                            with st.expander("Show content snippet"):
                                st.write(doc.page_content)

                            st.caption(f"Source: {os.path.basename(source)}")
                            
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")

if __name__ == "__main__":
    main()

