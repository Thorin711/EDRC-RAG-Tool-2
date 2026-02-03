import streamlit as st
import uuid
from qdrant_client.http.models import (
    PointStruct, 
    FieldCondition, 
    MatchText, 
    Filter
)

from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"
COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"

ALL_COLLECTIONS = [COLLECTION_FULL, COLLECTION_JOURNAL, COLLECTION_EDRC]

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

def admin_app():
    st.set_page_config(page_title="Admin Panel", page_icon="🔑", layout="wide")
    st.title("🔑 Document Metadata Editor (Multi-Update)")

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = [] 
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = COLLECTION_EDRC

    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect.")
        st.stop()

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
        
        if selected_collection_name == COLLECTION_FULL:
            vector_store = load_full_store(embeddings, QDRANT_URL, qdrant_api_key)
        elif selected_collection_name == COLLECTION_JOURNAL:
            vector_store = load_journal_store(embeddings, QDRANT_URL, qdrant_api_key)
        else:
            vector_store = load_edrc_store(embeddings, QDRANT_URL, qdrant_api_key)
            
        qdrant_client = vector_store.client
        st.info(f"Connected to collection: **{selected_collection_name}**")
    except Exception as e:
        st.error(f"Failed to load models or connect to Qdrant: {e}")
        st.stop()

    st.header("2. Find Document Chunks")
    search_query = st.text_input("Search for a document by its title:")
    
    if st.button("Find Documents"):
        st.session_state.search_results = None
        st.session_state.selected_points = []

        if search_query:
            with st.spinner("Searching by title..."):
                try:
                    title_filter = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.title",
                                match=MatchText(text=search_query)
                            )
                        ]
                    )
                    
                    search_results, _ = qdrant_client.scroll(
                        collection_name=selected_collection_name,
                        scroll_filter=title_filter,
                        limit=200,
                        with_payload=True
                    )
                    
                    st.session_state.search_results = search_results
                    
                    st.session_state.selected_points = search_results 
                    
                    if not search_results:
                        st.warning("No documents found with that title.")
                        
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter a search query.")

    if st.session_state.selected_points:
        selected_points = st.session_state.selected_points
        point_ids_to_update = [point.id for point in selected_points]
        
        # Get metadata from the first chunk to use everywhere
        first_point_payload = selected_points[0].payload
        current_meta = first_point_payload.get("metadata", {}).copy()
        current_title = current_meta.get('title', '')
        
        st.markdown(f"---")
        st.header(f"3. Actions for Document (Found {len(selected_points)} chunks)")
        st.markdown(f"**Title:** `{current_title}`")
        st.markdown(f"**Content Snippet (from first chunk):**\n```\n{first_point_payload.get('page_content', '')[:250]}...\n```")
        
        with st.expander("Show all chunk IDs to be affected"):
            st.json(point_ids_to_update)
        
        
        edit_tab, delete_tab = st.tabs(["Edit Metadata", "⛔ Delete Document"])

        with edit_tab:
            st.subheader("Update Metadata Fields")
            st.write("Changes here will apply to **all** selected chunks.")
            
            with st.form("edit_form"):
                new_title = st.text_input("Title", value=current_title)
                new_authors = st.text_input("Authors", value=current_meta.get('authors', ''))
                new_year = st.number_input("Year", min_value=0, max_value=2100, step=1, value=current_meta.get('year', 2024))
                new_doi = st.text_input("DOI", value=current_meta.get('doi', ''))
                
                st.markdown("---")
                
                apply_all_edit = st.checkbox(
                    "Apply these metadata changes to ALL collections (full_papers, journal_papers, edrc_papers)",
                    value=False,
                    help="If checked, this update will be applied to documents with the *original* title in all three collections."
                )
                
                submitted = st.form_submit_button(f"Save Changes", type="primary")

                if submitted:
                    
                    payload_to_merge = {
                        "metadata": {
                            "title": new_title,
                            "authors": new_authors,
                            "year": int(new_year),
                            "doi": new_doi
                        }
                    }

                    # Determine which collections to update
                    if apply_all_edit:
                        collections_to_update = ALL_COLLECTIONS
                        st.info("Applying changes to ALL collections...")
                    else:
                        collections_to_update = [selected_collection_name]
                        st.info(f"Applying changes to {selected_collection_name} only...")

                    with st.spinner(f"Saving changes..."):
                        try:
                            total_chunks_updated = 0
                            
                            for collection_name in collections_to_update:
                                title_filter = Filter(
                                    must=[
                                        FieldCondition(
                                            key="metadata.title",
                                            match=MatchText(text=current_title)
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
                                    st.write(f"ℹ️ No document matching '{current_title}' found in `{collection_name}`. Skipping.")
                                    continue
                                
                                # Apply the new payload to the found IDs
                                qdrant_client.set_payload(
                                    collection_name=collection_name,
                                    points=point_ids,  
                                    payload=payload_to_merge, # Use new metadata
                                    wait=True
                                )
                                st.write(f"✅ Updated {len(point_ids)} chunks in `{collection_name}`.")
                                total_chunks_updated += len(point_ids)

                            st.success(f"Metadata updated successfully for a total of {total_chunks_updated} chunks! 🎉")
                            st.balloons()
                            
                            # Clear state to be ready for the next search
                            st.session_state.search_results = None
                            st.session_state.selected_points = []
                            st.rerun() # Rerun to hide the form

                        except Exception as e:
                            st.error(f"An error occurred: {e}")


        with delete_tab:
            st.subheader("⛔ Danger Zone: Delete Document")
            st.warning(f"**WARNING:** You are about to permanently delete document chunks associated with this title. This action **cannot** be undone.")
            
            st.markdown("---")
            
            with st.form("delete_form"):
                confirm_check = st.checkbox(f"I understand I am permanently deleting chunks for '{current_title}'.")
                confirm_title = st.text_input(
                    "To confirm, please type the *exact* title of the document:", 
                    placeholder="Type title to confirm..."
                )
                
                st.markdown("---")
                
                apply_all_delete = st.checkbox(
                    "Permanently delete from ALL collections (full_papers, journal_papers, edrc_papers)",
                    value=False,
                    help="If checked, this will delete all chunks matching this title from all three collections."
                )
                
                st.markdown("---")
                
                submitted_delete = st.form_submit_button(
                    "DELETE DOCUMENT (PERMANENTLY)", 
                    type="primary", 
                    use_container_width=True
                )

                if submitted_delete:
                    is_confirmed = confirm_check and (confirm_title == current_title)

                    if is_confirmed:
                        
                        # Determine which collections to delete from
                        if apply_all_delete:
                            collections_to_delete_from = ALL_COLLECTIONS
                            st.info("Deleting from ALL collections...")
                        else:
                            collections_to_delete_from = [selected_collection_name]
                            st.info(f"Deleting from {selected_collection_name} only...")

                        with st.spinner(f"Deleting document chunks..."):
                            try:
                                total_chunks_deleted = 0
                                
                                # +++ LOOP THROUGH EACH COLLECTION +++
                                for collection_name in collections_to_delete_from:
                                    # Find the points in this collection by title
                                    title_filter = Filter(
                                        must=[
                                            FieldCondition(
                                                key="metadata.title",
                                                match=MatchText(text=current_title)
                                            )
                                        ]
                                    )
                                    points_to_delete, _ = qdrant_client.scroll(
                                        collection_name=collection_name,
                                        scroll_filter=title_filter,
                                        limit=500,
                                        with_payload=False
                                    )
                                    
                                    point_ids = [point.id for point in points_to_delete]
                                    
                                    if not point_ids:
                                        st.write(f"ℹ️ No document matching '{current_title}' found in `{collection_name}`. Skipping.")
                                        continue

                                    # Perform the delete
                                    qdrant_client.delete(
                                        collection_name=collection_name,
                                        points_selector=point_ids
                                    )
                                    st.write(f"🗑️ Deleted {len(point_ids)} chunks from `{collection_name}`.")
                                    total_chunks_deleted += len(point_ids)
                                
                                st.success(f"Successfully deleted a total of {total_chunks_deleted} chunks! 🗑️")
                                
                                # Clear state
                                st.session_state.search_results = None
                                st.session_state.selected_points = []
                                st.rerun()

                            except Exception as e:
                                st.error(f"An error occurred during deletion: {e}")
                    else:
                        st.error("Confirmation failed. Please check the box AND type the title correctly.")


if __name__ == "__main__":
    admin_app()