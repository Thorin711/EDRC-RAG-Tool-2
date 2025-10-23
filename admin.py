import streamlit as st
from qdrant_client.http.models import (
    PointStruct, Filter, FieldCondition, MatchText, 
    PayloadSchemaType # <-- Add this import
)
import uuid

# --- Add these imports from your main app.py ---
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

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = [] 

    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect.")
        st.stop()

    try:
        embeddings = load_embedding_model()
        vector_store = load_vector_store(
            embeddings,
            COLLECTION_EDRC,
            QDRANT_URL,
            qdrant_api_key
        )
        qdrant_client = vector_store.client
        st.info(f"Connected to collection: **{COLLECTION_EDRC}**")

        # --- START: NEW INDEX CREATION LOGIC ---
        # Check if the text index for 'metadata.title' exists
        collection_info = qdrant_client.get_collection(COLLECTION_EDRC)
        payload_schema = collection_info.payload_schema
        
        title_schema = payload_schema.get("metadata.title")
        
        # If schema doesn't exist or isn't of type TEXT, create it
        if not title_schema or title_schema.data_type != PayloadSchemaType.TEXT:
            with st.spinner("Creating text index for 'metadata.title'... This may take a moment."):
                qdrant_client.create_field_index(
                    collection_name=COLLECTION_EDRC,
                    field_name="metadata.title",
                    field_schema=PayloadSchemaType.TEXT
                )
            st.success("Text index for 'metadata.title' created! Please refresh the page to continue.")
            st.stop() # Stop the script. User needs to refresh.
        # --- END: NEW INDEX CREATION LOGIC ---

    except Exception as e:
        st.error(f"Failed to load models or connect to Qdrant: {e}")
        st.stop()

    # --- 1. Search Section (MODIFIED for Title Search) ---
    st.header("1. Find Document Chunks by Title")
    search_query = st.text_input("Search for a document by keyword in the title:")
    
    if st.button("Find Documents"):
        if search_query:
            with st.spinner("Searching by title..."):
                try:
                    # 1. Create a filter to match text
                    title_filter = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.title", 
                                match=MatchText(text=search_query) 
                            )
                        ]
                    )
                    
                    # 2. Use qdrant_client.scroll to get all results
                    search_results, _ = qdrant_client.scroll(
                        collection_name=COLLECTION_EDRC,
                        scroll_filter=title_filter,
                        limit=100, 
                        with_payload=True 
                    )
                    
                    st.session_state.search_results = search_results
                    st.session_state.selected_points = [] 
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter a search query.")

    # --- 2. Select Section (This section remains the same) ---
    if st.session_state.search_results:
        st.subheader(f"Found {len(st.session_state.search_results)} matching chunks")
        
        doc_map = {}
        for point in st.session_state.search_results:
            payload = point.payload
            metadata = payload.get("metadata", {})
            content = payload.get("page_content", "")
            point_id = point.id 
            title = metadata.get('title', 'No Title')
            
            label = f"'{title}' | Snippet: \"{content[:75]}...\" (ID: {point_id})"
            doc_map[label] = point

        selected_labels = st.multiselect(
            "Select all document chunks to update:",
            options=doc_map.keys(),
            help="You can select multiple snippets that belong to the same article."
        )

        if selected_labels:
            st.session_state.selected_points = [doc_map[label] for label in selected_labels]
        else:
            st.session_state.selected_points = []

    # --- 3. Edit Form Section (This section remains the same) ---
    if st.session_state.selected_points:
        st.header("2. Edit Metadata for Selected Chunks")
        
        selected_points = st.session_state.selected_points
        st.markdown(f"**You are about to edit {len(selected_points)} document chunks.**")
        
        point_ids_to_update = [point.id for point in selected_points]
        with st.expander("Show IDs to be updated"):
            st.json(point_ids_to_update)
        
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
                            collection_name=COLLECTION_EDRC,
                            points=point_ids_to_update,
                            payload=payload_to_merge,
                            wait=True
                        )
                        
                        st.success(f"Metadata updated successfully for {len(point_ids_to_update)} chunks! 🎉")
                        st.balloons()
                        
                        st.session_state.search_results = None
                        st.session_state.selected_points = []

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    admin_app()