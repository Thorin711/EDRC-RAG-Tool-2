# metadata_fixer.py
import streamlit as st
import uuid
from qdrant_client.http.models import (
    PointStruct, 
    FieldCondition, 
    MatchText, 
    Filter,
    Range  # Import Range for date filtering
)

from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants from your other scripts ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"
COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"

ALL_COLLECTIONS = [COLLECTION_FULL, COLLECTION_JOURNAL, COLLECTION_EDRC]

# --- Cache functions from admin.py ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

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

def fixer_app():
    st.set_page_config(page_title="Metadata Fixer", page_icon="🛠️", layout="wide")
    st.title("🛠️ Bulk Metadata Fixer (Year=0)")

    if "missing_data_files" not in st.session_state:
        st.session_state.missing_data_files = None
    if "scanned" not in st.session_state:
        st.session_state.scanned = False

    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect.")
        st.stop()

    try:
        # We only need to load one store to get the underlying client
        # The client can access ALL collections
        embeddings = load_embedding_model()
        vector_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
        qdrant_client = vector_store.client
        st.info(f"Connected to Qdrant cluster at {QDRANT_URL}")
    except Exception as e:
        st.error(f"Failed to load models or connect to Qdrant: {e}")
        st.stop()

    st.header("1. Find All Documents with Year = 0")
    
    if st.button("Scan All Collections for Missing Dates", type="primary"):
        st.session_state.scanned = True
        st.session_state.missing_data_files = {}
        master_file_list = {}

        with st.spinner("Scanning all collections... This may take a moment."):
            # Define a filter to find points where year is 0
            # We use lte=0 to also catch any erroneous negative numbers
            year_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.year",
                        range=Range(lte=0) # Find all documents with year 0 or less
                    )
                ]
            )
            
            for collection_name in ALL_COLLECTIONS:
                st.write(f"Scanning `{collection_name}`...")
                try:
                    # Scroll through all points matching the filter
                    # Increase limit if you have > 5000 chunks with year=0
                    results, _ = qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=year_filter,
                        limit=5000, 
                        with_payload=True
                    )
                    
                    # Group chunks by document title
                    for point in results:
                        meta = point.payload.get("metadata", {})
                        # Use the original title as the key
                        original_title = meta.get("title", f"Unknown_ID_{point.id}")
                        
                        if original_title not in master_file_list:
                            # This is the first time we see this document
                            master_file_list[original_title] = {
                                "metadata": meta,
                                "collections": {collection_name} # Use a set to store collections
                            }
                        else:
                            # Add this collection to the set of places this doc was found
                            master_file_list[original_title]["collections"].add(collection_name)
                            
                except Exception as e:
                    st.error(f"Error scanning `{collection_name}`: {e}")
        
        st.session_state.missing_data_files = master_file_list
        st.success(f"Scan complete! Found {len(master_file_list)} documents with missing year.")
        st.rerun() # Rerun to display the results

    # --- Display the "To-Do" List ---
    if st.session_state.scanned:
        if not st.session_state.missing_data_files:
            st.info("No documents with missing dates (Year=0) were found.")
        else:
            st.markdown("---")
            st.header(f"To-Do List: {len(st.session_state.missing_data_files)} Documents")
            
            # Create a copy for safe iteration, as we'll be deleting items
            files_to_fix = dict(st.session_state.missing_data_files)
            
            # The 'title' here is the ORIGINAL title, used as the key
            for original_title, data in files_to_fix.items():
                with st.container(border=True):
                    current_meta = data['metadata']
                    collections_found_in = list(data['collections'])
                    
                    st.info(f"Found in: `{', '.join(collections_found_in)}`")
                    
                    # Use the form logic directly from admin.py
                    with st.form(key=f"form_{original_title}"):
                        
                        # --- MODIFICATION ---
                        # Add a field to edit the title
                        new_title = st.text_input("Title", value=original_title)
                        # --- END MODIFICATION ---

                        new_year = st.number_input(
                            "Year", 
                            min_value=0, 
                            max_value=2100, 
                            step=1, 
                            value=0 # Default to a sensible year
                        )
                        new_authors = st.text_input("Authors", value=current_meta.get('authors', ''))
                        new_doi = st.text_input("DOI", value=current_meta.get('doi', ''))
                        
                        submitted = st.form_submit_button(
                            f"Save Changes to {len(collections_found_in)} Collection(s)", 
                            type="primary"
                        )

                        if submitted:
                            payload_to_merge = {
                                "metadata": {
                                    "title": new_title, # Use the new title from the form
                                    "authors": new_authors,
                                    "year": int(new_year),
                                    "doi": new_doi
                                }
                            }

                            with st.spinner(f"Saving changes for '{original_title}'..."):
                                try:
                                    total_chunks_updated = 0
                                    
                                    for collection_name in collections_found_in:
                                        # Find all points in this collection matching the ORIGINAL title
                                        title_filter = Filter(
                                            must=[
                                                FieldCondition(
                                                    key="metadata.title",
                                                    match=MatchText(text=original_title) # Find by old title
                                                )
                                            ]
                                        )
                                        points_to_update, _ = qdrant_client.scroll(
                                            collection_name=collection_name,
                                            scroll_filter=title_filter,
                                            limit=500,
                                            with_payload=False
                                        )
                                        
                                        point_ids = [point.id for point in points_to_update]
                                        
                                        if not point_ids:
                                            st.write(f"ℹ️ No matching chunks found in `{collection_name}`. Skipping.")
                                            continue
                                        
                                        # Apply the new payload (with the new title)
                                        qdrant_client.set_payload(
                                            collection_name=collection_name,
                                            points=point_ids,  
                                            payload=payload_to_merge,
                                            wait=True
                                        )
                                        st.write(f"✅ Updated {len(point_ids)} chunks in `{collection_name}`.")
                                        total_chunks_updated += len(point_ids)

                                    st.success(f"Metadata updated successfully for {total_chunks_updated} chunks! 🎉")
                                    
                                    # "Tick off" the item by removing it from the session state
                                    # Use the original_title key
                                    del st.session_state.missing_data_files[original_title]
                                    st.rerun() # Rerun to refresh the list

                                except Exception as e:
                                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    fixer_app()