import streamlit as st
from qdrant_client.http.models import PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
import uuid



# --- CONFIGURATION (Copied from app.py) ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io"
COLLECTION_FULL = "full_papers"
COLLECTION_JOURNAL = "journal_papers"
COLLECTION_EDRC = "edrc_papers"

# --- CACHING FUNCTIONS (Copied from app.py) ---
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
        _api_key (str): The API key for the Qdrant Cloud instance.from langchain_huggingface import HuggingFaceEmbeddings

    Returns:
        langchain_qdrant.Qdrant: The loaded vector store instance.
    """

    # --- START: MODIFIED SECTION ---
    # These keys must match the payload structure from your migrator script
    # {"page_content": "...", "metadata": {...}}

    return Qdrant.from_existing_collection(
        embedding=_embeddings,
        collection_name=_collection_name,
        url=_url,
        api_key=_api_key,

        # The key holding the main text
        content_payload_key="page_content", 

        # The key holding the nested metadata dictionary
        metadata_payload_key="metadata"
    )
    # --- END: MODIFIED SECTION ---

# --- Main Admin App ---
def admin_app():
    st.set_page_config(page_title="Admin Panel", page_icon="🔑", layout="wide")
    st.title("🔑 Document Metadata Editor")

    # --- Initialize Session State ---
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "selected_doc_id" not in st.session_state:
        st.session_state.selected_doc_id = None

    # --- Load Models and DB Client ---
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot connect.")
        st.stop()

    try:
        embeddings = load_embedding_model()
        vector_store = load_vector_store(
            embeddings,
            COLLECTION_EDRC,  # Hardcoding to EDRC collection
            QDRANT_URL,
            qdrant_api_key
        )
        qdrant_client = vector_store.client
        st.info(f"Connected to collection: **{COLLECTION_EDRC}**")
    except Exception as e:
        st.error(f"Failed to load models or connect to Qdrant: {e}")
        st.stop()

    # --- 1. Search Section ---
    st.header("1. Find Document to Edit")
    search_query = st.text_input("Search for a document by title or topic:")
    
    if st.button("Find Documents"):
        if search_query:
            with st.spinner("Searching..."):
                # We use similarity_search_with_score to ensure we get the Point ID
                results = vector_store.similarity_search_with_score(search_query, k=5)
                st.session_state.search_results = results
                st.session_state.selected_doc = None # Clear previous selection
        else:
            st.warning("Please enter a search query.")

    # --- 2. Select Section ---
    if st.session_state.search_results:
        st.subheader("Search Results")
        
        # Create labels for the radio buttons
        doc_options = []
        for doc, score in st.session_state.search_results:
            title = doc.metadata.get('title', 'No Title')
            point_id = doc.metadata.get('id', 'MISSING_ID') # LangChain adds the ID here
            doc_options.append(f"'{title}' (ID: {point_id})")

        selected_option = st.radio("Select a document to edit:", doc_options, index=None)

        if selected_option:
            # Find the index of the selected option
            selected_index = doc_options.index(selected_option)
            # Get the Document object and its ID
            doc, _ = st.session_state.search_results[selected_index]
            point_id = doc.metadata.get('id')
            
            # Store them in session state for the form
            st.session_state.selected_doc = doc
            st.session_state.selected_doc_id = point_id

    # --- 3. Edit Form Section ---
    if st.session_state.selected_doc:
        st.header("2. Edit Metadata")
        
        doc = st.session_state.selected_doc
        point_id = st.session_state.selected_doc_id
        
        st.markdown(f"**Editing Document:** `{doc.metadata.get('title', 'No Title')}`")
        st.caption(f"**Point ID:** `{point_id}`")
        st.markdown(f"**Content Snippet:**\n```\n{doc.page_content[:250]}...\n```")

        with st.form("edit_form"):
            st.subheader("Update Fields")
            
            # --- FIX: THIS LINE MUST COME FIRST ---
            # Get current metadata
            current_meta = doc.metadata.copy()
            
            # --- THEN, CREATE THE FORM FIELDS ---
            new_title = st.text_input("Title", value=current_meta.get('title', ''))
            new_authors = st.text_input("Authors", value=current_meta.get('authors', ''))
            new_year = st.number_input("Year", min_value=0, max_value=2100, step=1, value=current_meta.get('year', 2024))
            new_doi = st.text_input("DOI", value=current_meta.get('doi', ''))
            
            # --- THEN, CREATE THE SUBMIT BUTTON ---
            submitted = st.form_submit_button("Save Changes to Database", type="primary")

            # --- FINALLY, HANDLE THE SUBMISSION ---
            if submitted:
                if not point_id:
                    st.error("Error: Point ID is missing. Cannot update.")
                else:
                    with st.spinner("Saving changes..."):
                        try:
                            # 1. Create the new metadata dictionary
                            updated_metadata = current_meta.copy()
                            
                            # Remove the 'id' key - it's not part of the payload
                            if 'id' in updated_metadata:
                                del updated_metadata['id'] 

                            # 2. Apply the changes from the form
                            updated_metadata['title'] = new_title
                            updated_metadata['authors'] = new_authors
                            updated_metadata['year'] = int(new_year)
                            updated_metadata['doi'] = new_doi
                            
                            # 3. Create the final payload to merge
                            new_payload = {
                                "metadata": updated_metadata
                            }

                            # 4. Use set_payload to update the point
                            qdrant_client.set_payload(
                                collection_name=COLLECTION_EDRC,
                                points=[point_id],  # The ID of the point to update
                                payload=new_payload, # The new data to merge
                                wait=True
                            )
                            
                            st.success("Metadata updated successfully! 🎉")
                            st.balloons()
                            
                            # Clear state to prevent re-submission
                            st.session_state.search_results = None
                            st.session_state.selected_doc = None
                            st.session_state.selected_doc_id = None

                        except Exception as e:
                            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    admin_app()