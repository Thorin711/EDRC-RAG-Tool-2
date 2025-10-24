import streamlit as st
import uuid
# --- Cleaned up Qdrant Imports ---
from qdrant_client.http.models import (
    PointStruct, 
    FieldCondition, 
    MatchText, 
    Filter
)

# --- Imports from your main app.py ---
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION (Copied from app.py) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"
COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"

# --- CACHING FUNCTIONS (Copied from app.py) ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_vector_store(_embeddings, _collection_name, _url, _api_key):
    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=_collection_name,
        url=_url,
        api_key=_api_key,
        content_payload_key="page_content",
        metadata_payload_key="metadata"
    )

# --- Main Admin App ---
def admin_app():
    st.set_page_config(page_title="Admin Panel", page_icon="🔑", layout="wide")
    st.title("🔑 Document Metadata Editor (Multi-Update)")

    # --- Initialize Session State ---
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = [] 
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = COLLECTION_EDRC

    # --- Load Models and DB Client ---
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect.")
        st.stop()

    # --- 1. Database Selection ---
    st.header("1. Select Database")
    DB_OPTIONS = {
        "Full Database": COLLECTION_FULL,
        "Journal Articles Only": COLLECTION_JOURNAL,
        "EDRC Only": COLLECTION_EDRC,
    }
    
    current_collection_index = list(DB_OPTIONS.values()).index(st.session_state.selected_collection)
    
    db_choice = st.radio(
        "Select database to edit:",
        options=DB_OPTIONS.keys(),
        horizontal=True,
        index=current_collection_index
    )
    
    selected_collection_name = DB_OPTIONS[db_choice]

    if selected_collection_name != st.session_state.selected_collection:
        st.session_state.selected_collection = selected_collection_name
        st.session_state.search_results = None
        st.session_state.selected_points = []
        st.rerun()

    try:
        embeddings = load_embedding_model()
        vector_store = load_vector_store(
            embeddings,
            selected_collection_name,
            QDRANT_URL,
            qdrant_api_key
        )
        qdrant_client = vector_store.client
        st.info(f"Connected to collection: **{selected_collection_name}**")
    except Exception as e:
        st.error(f"Failed to load models or connect to Qdrant: {e}")
        st.stop()

    # --- 2. Find Document Chunks to Edit (MODIFIED) ---
    st.header("2. Find Document Chunks to Edit")
    search_query = st.text_input("Search for a document by its title:")
    
    if st.button("Find Documents"):
        # Clear previous results on a new search
        st.session_state.search_results = None
        st.session_state.selected_points = []

        if search_query:
            with st.spinner("Searching by title..."):
                try:
                    # 1. Define a filter for the metadata title
                    title_filter = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.title",
                                match=MatchText(text=search_query)
                            )
                        ]
                    )
                    
                    # 2. Use client.scroll() to get all matching points
                    search_results, _ = qdrant_client.scroll(
                        collection_name=selected_collection_name,
                        scroll_filter=title_filter,
                        limit=200,  # Set a high limit to get all chunks
                        with_payload=True
                    )
                    
                    st.session_state.search_results = search_results # Store for reference
                    
                    # --- NEW LOGIC ---
                    # Automatically select all found points for editing
                    st.session_state.selected_points = search_results 
                    
                    if not search_results:
                        st.warning("No documents found with that title.")
                        
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter a search query.")

    # --- 3. Edit Metadata (Previously Section 4) ---
    st.header("3. Edit Metadata")
    if st.session_state.selected_points:
        selected_points = st.session_state.selected_points
        st.markdown(f"**Found {len(selected_points)} document chunks for this title. You are about to edit all of them.**")
        
        point_ids_to_update = [point.id for point in selected_points]
        with st.expander("Show IDs to be updated"):
            st.json(point_ids_to_update)
        
        # Get metadata from the first chunk to pre-fill the form
        first_point_payload = selected_points[0].payload
        current_meta = first_point_payload.get("metadata", {}).copy()
        
        st.markdown(f"**Content Snippet (from first selected item):**\n```\n{first_point_payload.get('page_content', '')[:250]}...\n```")

        with st.form("edit_form"):
            st.subheader("Update Fields (will apply to all selected items)")
            
            new_title = st.text_input("Title", value=current_meta.get('title', ''))
            new_authors = st.text_input("Authors", value=current_meta.get('authors', ''))
            new_year = st.number_input("Year", min_value=0, max_value=2100, step=1, value=current_meta.get('year', 2024))
            new_doi = st.text_input("DOI", value=current_meta.get('doi', ''))
            
            submitted = st.form_submit_button(f"Save Changes to {len(selected_points)} Chunks", type="primary")

            if submitted:
                with st.spinner(f"Saving changes to {len(point_ids_to_update)} chunks..."):
                    try:
                        payload_to_merge = {
                            "metadata": {
                                "title": new_title,
                                "authors": new_authors,
                                "year": int(new_year),
                                "doi": new_doi
                            }
                        }

                        qdrant_client.set_payload(
                            collection_name=selected_collection_name,
                            points=point_ids_to_update,  
                            payload=payload_to_merge,   
                            wait=True
                        )
                        
                        st.success(f"Metadata updated successfully for {len(point_ids_to_update)} chunks! 🎉")
                        st.balloons()
                        
                        # Clear state to be ready for the next search
                        st.session_state.search_results = None
                        st.session_state.selected_points = []
                        st.rerun() # Rerun to hide the form

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    admin_app()