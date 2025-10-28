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

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"
COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"

# --- CACHING ---
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

def admin_app():
    st.set_page_config(page_title="Admin Panel", page_icon="🔑", layout="wide")
    st.title("🔑 Document Metadata Editor (Multi-Update)")

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = [] 
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = COLLECTION_FULL

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

    # --- Section 3: Edit or Delete (NEW TABS) ---
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

        # --- EDIT TAB (Existing Logic) ---
        with edit_tab:
            st.subheader("Update Metadata Fields")
            st.write("Changes here will apply to **all** selected chunks.")
            
            with st.form("edit_form"):
                new_title = st.text_input("Title", value=current_title)
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


        # --- DELETE TAB ---
        with delete_tab:
            st.subheader("⛔ Danger Zone: Delete Document")
            st.warning(f"**WARNING:** You are about to permanently delete **{len(selected_points)}** document chunks associated with this title. This action **cannot** be undone.")
            
            st.markdown("---")
            
            # Wrap all interactive elements in a form
            with st.form("delete_form"):
                confirm_check = st.checkbox(f"I understand I am permanently deleting {len(selected_points)} chunks.")
                confirm_title = st.text_input(
                    "To confirm, please type the *exact* title of the document:", 
                    placeholder="Type title to confirm..."
                )
                
                st.markdown("---")
                
                # Change the button to a form_submit_button
                submitted_delete = st.form_submit_button(
                    "DELETE DOCUMENT (PERMANENTLY)", 
                    type="primary", 
                    use_container_width=True
                )

                if submitted_delete:
                    # Move the confirmation check to *after* the button is pressed
                    is_confirmed = confirm_check and (confirm_title == current_title)

                    if is_confirmed:
                        with st.spinner(f"Deleting {len(point_ids_to_update)} chunks..."):
                            try:
                                qdrant_client.delete(
                                    collection_name=selected_collection_name,
                                    points_selector=point_ids_to_update
                                )
                                
                                st.success(f"Successfully deleted {len(point_ids_to_update)} chunks! 🗑️")
                                
                                # Clear state
                                st.session_state.search_results = None
                                st.session_state.selected_points = []
                                st.rerun()

                            except Exception as e:
                                st.error(f"An error occurred during deletion: {e}")
                    else:
                        # If confirmation fails, show an error *inside* the form
                        st.error("Confirmation failed. Please check the box AND type the title correctly.")


if __name__ == "__main__":
    admin_app()