import streamlit as st
from qdrant_client.http.models import (
    FieldCondition,
    MatchText,
    Filter
)

from common import (
    get_secret,
    get_qdrant_url,
    load_embedding_model,
    load_store,
    build_exact_document_filter,
    scroll_all,
    COLLECTION_EDRC,
    ALL_COLLECTIONS,
    DB_OPTIONS,
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
    if "pending_delete" not in st.session_state:
        st.session_state.pending_delete = None

    qdrant_api_key = get_secret("QDRANT_API_KEY")
    qdrant_url = get_qdrant_url()
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in secrets (Streamlit or Env). App cannot connect.")
        st.stop()

    st.header("1. Select Database")

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
        st.session_state.pending_delete = None
        st.rerun()

    try:
        embeddings = load_embedding_model()
        vector_store = load_store(embeddings, selected_collection_name, qdrant_url, qdrant_api_key)
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
        st.session_state.pending_delete = None

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
                    "Apply these metadata changes to ALL collections (full_papers_v2, journal_papers_v2, edrc_papers_v2)",
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

                    # Exact match on doc_id (or, for documents uploaded before
                    # doc_id existed, on the ORIGINAL title) -- never a
                    # substring match, so this can't sweep in a different
                    # document with a similar title.
                    exact_filter = build_exact_document_filter(current_meta)

                    with st.spinner(f"Saving changes..."):
                        try:
                            total_chunks_updated = 0

                            for collection_name in collections_to_update:
                                points_to_update = scroll_all(
                                    qdrant_client,
                                    collection_name,
                                    exact_filter,
                                    with_payload=False,
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
                            st.session_state.pending_delete = None
                            st.rerun() # Rerun to hide the form

                        except Exception as e:
                            st.error(f"An error occurred: {e}")


        with delete_tab:
            st.subheader("⛔ Danger Zone: Delete Document")
            st.warning(f"**WARNING:** You are about to permanently delete document chunks associated with this title. This action **cannot** be undone.")

            st.markdown("---")

            # Unique key for the document currently selected in the UI, so a
            # stale preview from a previous document can never be confirmed
            # against this one.
            doc_key = current_meta.get("doc_id") or current_title

            with st.form("delete_form"):
                confirm_check = st.checkbox(f"I understand I am permanently deleting chunks for '{current_title}'.")
                confirm_title = st.text_input(
                    "To confirm, please type the *exact* title of the document:",
                    placeholder="Type title to confirm..."
                )

                st.markdown("---")

                apply_all_delete = st.checkbox(
                    "Permanently delete from ALL collections (full_papers_v2, journal_papers_v2, edrc_papers_v2)",
                    value=False,
                    help="If checked, this will delete all chunks matching this title from all three collections."
                )

                st.markdown("---")

                submitted_delete = st.form_submit_button(
                    "Preview Deletion",
                    type="primary",
                    use_container_width=True
                )

                if submitted_delete:
                    is_confirmed = confirm_check and (confirm_title == current_title)

                    if is_confirmed:
                        # Exact match on doc_id (or, for documents uploaded
                        # before doc_id existed, on the ORIGINAL title) --
                        # never a substring match, so this can't sweep in a
                        # different document with a similar title.
                        exact_filter = build_exact_document_filter(current_meta)
                        collections_to_delete_from = ALL_COLLECTIONS if apply_all_delete else [selected_collection_name]

                        with st.spinner("Scanning for chunks to delete..."):
                            preview = {}
                            for collection_name in collections_to_delete_from:
                                points = scroll_all(qdrant_client, collection_name, exact_filter, with_payload=True)
                                if points:
                                    preview[collection_name] = points

                        st.session_state.pending_delete = {"doc_key": doc_key, "preview": preview}
                        st.rerun()
                    else:
                        st.error("Confirmation failed. Please check the box AND type the title correctly.")

            pending = st.session_state.get("pending_delete")
            if pending and pending.get("doc_key") == doc_key:
                preview = pending["preview"]
                total = sum(len(points) for points in preview.values())

                if total == 0:
                    st.info("No matching chunks were found in the target collection(s) -- nothing to delete.")
                    if st.button("Dismiss", key="dismiss_empty_delete_preview"):
                        st.session_state.pending_delete = None
                        st.rerun()
                else:
                    st.markdown("#### Confirm exactly what will be deleted")
                    distinct_titles = set()
                    for collection_name, points in preview.items():
                        titles_here = {
                            p.payload.get("metadata", {}).get("title", "(no title)") for p in points
                        }
                        distinct_titles |= titles_here
                        st.write(f"- `{collection_name}`: **{len(points)}** chunk(s)")

                    if len(distinct_titles) > 1:
                        st.error(
                            "⚠️ This selector matched more than one distinct title: "
                            + ", ".join(f"'{t}'" for t in distinct_titles)
                            + ". Refusing to proceed -- please investigate before deleting."
                        )
                    else:
                        st.write(f"**Total: {total} chunk(s) across {len(preview)} collection(s).**")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            really_delete = st.button(
                                f"⛔ Yes, permanently delete {total} chunk(s)",
                                type="primary",
                                use_container_width=True,
                            )
                        with col_cancel:
                            cancel_delete = st.button("Cancel", use_container_width=True)

                        if cancel_delete:
                            st.session_state.pending_delete = None
                            st.rerun()

                        if really_delete:
                            with st.spinner("Deleting document chunks..."):
                                try:
                                    total_chunks_deleted = 0
                                    for collection_name, points in preview.items():
                                        point_ids = [p.id for p in points]
                                        qdrant_client.delete(
                                            collection_name=collection_name,
                                            points_selector=point_ids
                                        )
                                        st.write(f"🗑️ Deleted {len(point_ids)} chunks from `{collection_name}`.")
                                        total_chunks_deleted += len(point_ids)

                                    st.success(f"Successfully deleted a total of {total_chunks_deleted} chunks! 🗑️")

                                    # Clear state
                                    st.session_state.pending_delete = None
                                    st.session_state.search_results = None
                                    st.session_state.selected_points = []
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"An error occurred during deletion: {e}")


if __name__ == "__main__":
    admin_app()