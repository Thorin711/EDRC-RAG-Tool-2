# -*- coding: utf-8 -*-
"""
Streamlit App to process PDFs using a personal GROBID server,
review and EDIT the extracted YAML/text, chunk the content,
and upload the final chunks to a Qdrant vector database.
"""

import streamlit as st
import requests
import re
import os
import time
from bs4 import BeautifulSoup
import unicodedata

# --- NEW IMPORTS ---
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
import qdrant_client # To ensure it's installed

# --- CONFIGURATION ---

# 1. GROBID SERVER
GROBID_API_URL = "https://thorin711-edrc-grobid.hf.space/api/processFulltextDocument"
REQUEST_TIMEOUT = 180
MAX_RETRIES = 2
RETRY_DELAY = 5

# 2. EMBEDDING & VECTOR DB CONFIG
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- QDRANT CONFIG ---
QDRANT_URL = "https://ba7e46f3-88ed-4d8b-99ed-8302a2d4095f.eu-west-2-0.aws.cloud.qdrant.io" 

# These must match the collection names in your RAG app
DB_OPTIONS = {
    "Full Database": "full_papers",
    "Journal Articles Only": "journal_papers",
    "EDRC Only": "edrc_papers",
}
# --- -------------------- ---


# --- 1. GROBID API CALL LOGIC (Unchanged) ---

def sanitize_filename(filename):
    """
    Cleans a filename by removing special characters, normalizing,
    and ensuring a basic name if all chars are stripped.
    """
    try:
        normalized_name = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
        safe_name = re.sub(r'[^\w.-]', '', normalized_name.replace(' ', '-'))
        if not safe_name:
            return "document.pdf"
        return safe_name
    except Exception as e:
        print(f"Error sanitizing filename: {e}")
        return "document.pdf"

def call_grobid_api(pdf_bytes, filename):
    """
    Calls the configured GROBID API to process a PDF.
    """
    if GROBID_API_URL.startswith("https://YOUR-HF-USERNAME"):
        st.error("Please update the `GROBID_API_URL` with your HF Space URL.", icon="🚨")
        return None, "Configuration Error"

    safe_filename = sanitize_filename(filename)
    multipart_payload = {
        'input': (safe_filename, pdf_bytes, 'application/pdf'),
        'consolidateHeader': (None, '1'),
        'consolidateCitations': (None, '0'),
    }
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            st.write(f"Connecting to server (Attempt {attempt + 1}/{MAX_RETRIES + 1})...")
            response = requests.post(
                GROBID_API_URL,
                files=multipart_payload, 
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.text, None
            else:
                error_message = f"API error (Status {response.status_code}): {response.text[:500]}"
                if attempt == MAX_RETRIES: return None, error_message
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES: return None, "Request timed out."
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES: return None, f"Connection error: {e}"
        st.write(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    return None, "Unknown error after all retries."

# --- 2. XML TO MARKDOWN LOGIC (Unchanged) ---

def parse_xml_to_markdown(xml_content, filename_for_download):
    """
    Parses GROBID's XML, extracts metadata and body.
    Returns: (title, authors, doi, year, body_str, download_filename)
    """
    try:
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        unwanted_section_headings = [
            "references", "bibliography", "acknowledgement", "acknowledgements",
            "declaration of competing interest", "credit authorship contribution statement"
        ]
        for div in soup.find_all('div'):
            head = div.find('head')
            if head and head.get_text(strip=True).lower() in unwanted_section_headings:
                div.decompose()

        title = "No Title Found"
        authors = []
        doi = ""
        year = ""
        abstract_text = ""

        title_stmt = soup.find('titleStmt')
        if title_stmt:
            title_tag = title_stmt.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        
        analytic_section = soup.find('analytic')
        if analytic_section:
            for author in analytic_section.find_all('author'):
                pers_name = author.find('persName')
                if pers_name:
                    forenames = " ".join([fn.get_text(strip=True) for fn in pers_name.find_all('forename')])
                    surname_tag = pers_name.find('surname')
                    surname = surname_tag.get_text(strip=True) if surname_tag else ""
                    authors.append(f"{forenames} {surname}".strip())
        
        doi_tag = soup.find('idno', type='DOI')
        if doi_tag:
            doi = doi_tag.get_text(strip=True)
            
        abstract_tag = soup.find('abstract')
        if abstract_tag:
            abstract_text = "\n".join([p.get_text(strip=True) for p in abstract_tag.find_all('p')])

        date_source = soup.find('publicationStmt') or soup.find('monogr')
        if date_source:
            date_tag = date_source.find('date')
            if date_tag:
                if date_tag.get('when'):
                    year = date_tag['when'][:4]
                else:
                    year_match = re.search(r'\b\d{4}\b', date_tag.get_text(strip=True))
                    if year_match:
                        year = year_match.group(0)

        markdown_body_parts = []
        if abstract_text:
            markdown_body_parts.append("## Abstract")
            markdown_body_parts.append(abstract_text)
        body_tag = soup.find('body')
        if body_tag:
            for section in body_tag.find_all('div', recursive=False):
                heading_tag = section.find('head')
                if heading_tag:
                    heading_text = heading_tag.get_text(strip=True)
                    n_attr = heading_tag.get('n', '')
                    heading_level = n_attr.count('.') + 2 if n_attr else 2
                    heading_prefix = '#' * heading_level
                    markdown_body_parts.append(f"\n{heading_prefix} {heading_text}")
                for p in section.find_all('p', recursive=False):
                    markdown_body_parts.append(p.get_text(strip=True))
        markdown_body = "\n\n".join(markdown_body_parts)
        base_name = os.path.splitext(filename_for_download)[0]
        download_filename = f"{base_name}.md"
        
        return title, authors, doi, year, markdown_body, download_filename

    except Exception as e:
        st.exception(e)
        return None, None, None, None, None, f"Error parsing XML: {e}"

# --- 3. NEW: CHUNKING & UPLOADING LOGIC ---

@st.cache_resource
def load_embedding_model():
    """Loads and caches the embedding model."""
    st.write(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    st.write("Embedding model loaded.")
    return model

def chunk_document(markdown_content, doc_metadata):
    """
    Splits the markdown document based on headers and applies metadata.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        return_each_line=False
    )
    
    # Split the content
    chunks = markdown_splitter.split_text(markdown_content)
    
    # Apply the base document metadata to all chunks
    for chunk in chunks:
        chunk.metadata.update(doc_metadata)
        
    st.write(f"Document split into {len(chunks)} chunks.")
    return chunks

def upload_chunks(chunks, embedding_model, url, api_key, collection_name):
    """
    Uploads document chunks to the Qdrant vector database.
    """
    if not chunks:
        st.warning("No chunks were created. Skipping upload.")
        return

    st.write(f"Uploading {len(chunks)} chunks to Qdrant collection: '{collection_name}'...")
    
    # This will add to the collection. 
    # Set force_recreate=True if you want to wipe the collection first.
    Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        prefer_grpc=True,
        force_recreate=False 
    )
    st.success("Upload to Qdrant complete!")

# --- 4. STREAMLIT APP UI (Modified) ---

def main():
    st.set_page_config(layout="wide", page_title="PDF to Vector DB Uploader")
    st.title("📄 PDF to Vector DB Uploader")
    st.markdown("Extract, review, and upload academic papers directly to your Qdrant database.")

    # --- NEW: Load API Key and Embedding Model ---
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    if not qdrant_api_key:
        st.error("`QDRANT_API_KEY` not found in Streamlit secrets. App cannot upload.", icon="🚨")
        st.stop()
        
    if QDRANT_URL == "https://YOUR-QDRANT-CLOUD-URL.com":
        st.error("Please update the `QDRANT_URL` variable in the script.", icon="🚨")
        st.stop()

    try:
        embeddings = load_embedding_model()
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

    # --- NEW: Collection Selection Radio Buttons ---
    st.subheader("Target Vector Collection")
    db_choice_key = st.radio(
        "Select the Qdrant collection to upload this document to:",
        options=DB_OPTIONS.keys(),
        horizontal=True,
    )
    selected_collection_name = DB_OPTIONS[db_choice_key]
    st.markdown("---")


    # --- File Uploader (Unchanged) ---
    uploaded_files = st.file_uploader(
        "Upload your PDF files", 
        type="pdf", 
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload one or more PDFs to begin processing.")
        st.markdown("---")
        st.subheader("Your Server:")
        st.code(GROBID_API_URL, language="text")
        return

    st.markdown("---")
    
    for uploaded_file in uploaded_files:
        unique_key = uploaded_file.file_id 
        st.header(f"Processing: `{uploaded_file.name}`")
        
        with st.spinner(f"Contacting server... This can take 1-3 minutes if it's waking up."):
            pdf_bytes = uploaded_file.getvalue()
            xml_result, error_msg = call_grobid_api(pdf_bytes, uploaded_file.name)

        if error_msg:
            st.error(f"**Failed to process `{uploaded_file.name}`:**\n\n{error_msg}")
            continue

        st.success(f"Successfully processed `{uploaded_file.name}`!")

        with st.spinner("Parsing XML and building Markdown..."):
            title, authors, doi, year, md_body, dl_filename = parse_xml_to_markdown(
                xml_result, uploaded_file.name
            )
        
        if md_body is None:
            st.error(f"Failed to parse XML for `{uploaded_file.name}`: {dl_filename}")
            continue

        # --- Display Results (Unchanged) ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("1. Review Metadata")
            edited_title = st.text_input("Title:", value=title, key=f"title_{unique_key}")
            edited_doi = st.text_input("DOI:", value=doi, key=f"doi_{unique_key}")
            edited_year = st.text_input("Year:", value=year, key=f"year_{unique_key}")
            
            if authors:
                author_list_str = "\n".join([f'  - "{auth}"' for auth in authors])
            else:
                author_list_str = "  - Not Available"
            
            edited_authors_str = st.text_area(
                "Authors (YAML list format):",
                value=author_list_str,
                height=150,
                key=f"authors_{unique_key}",
                help="Keep this in the YAML list format (e.g., '  - \"First Last\"')"
            )
            
        with col2:
            st.subheader("2. Review Body Text")
            edited_body = st.text_area(
                "Review the full extracted text:",
                value=md_body,
                height=500,
                key=f"body_{unique_key}"
            )

        # --- 3. Actions Column (Modified) ---
        with col1:
            st.subheader("3. Finalize & Export")
            st.write("") # Spacer
            
            # --- Assemble Final Content (for both buttons) ---
            safe_title = edited_title.replace('"', '\\"')
            safe_doi = edited_doi.replace('"', '\\"')
            safe_year = edited_year.replace('"', '\\"')

            final_yaml_str = "---\n"
            final_yaml_str += f'title: "{safe_title}"\n'
            final_yaml_str += "authors:\n"
            final_yaml_str += edited_authors_str + "\n"
            final_yaml_str += f'doi: "{safe_doi}"\n'
            final_yaml_str += f'year: "{safe_year}"\n'
            final_yaml_str += "---\n\n"
            
            final_markdown_for_download = final_yaml_str + edited_body
            
            # --- Download Button (Unchanged) ---
            st.download_button(
                label=f"Download `{dl_filename}`",
                data=final_markdown_for_download,
                file_name=dl_filename,
                mime="text/markdown",
                key=f"dl_{unique_key}"
            )

            st.write("") # Spacer

            # --- NEW: UPLOAD BUTTON ---
            if st.button("Confirm & Upload to Vector DB", type="primary", key=f"upload_{unique_key}"):
                with st.spinner(f"Uploading `{edited_title}` to Qdrant collection '{selected_collection_name}'..."):
                    try:
                        # 1. Create the metadata dictionary for the chunks
                        # Parse authors string back into a simple comma-separated string
                        authors_list = [
                            line.strip().lstrip('-').lstrip().strip('"') 
                            for line in edited_authors_str.split('\n') 
                            if line.strip() and line.strip() != "-"
                        ]
                        authors_string = ", ".join(authors_list)
                        
                        doc_metadata = {
                            "title": edited_title,
                            "authors": authors_string,
                            "doi": edited_doi,
                            "source": uploaded_file.name # Use the original PDF name as source
                        }
                        
                        # Add year only if it's a valid integer
                        try:
                            doc_metadata["year"] = int(edited_year)
                        except ValueError:
                            st.warning(f"Year '{edited_year}' is not a valid integer. Skipping 'year' metadata.")
                            pass

                        # 2. Chunk the document (using the full text, including YAML)
                        # The chunker will use headers but the YAML will be in the first chunk.
                        # This is fine as the metadata is applied to all chunks anyway.
                        chunks = chunk_document(final_markdown_for_download, doc_metadata)
                        
                        # 3. Upload the chunks
                        upload_chunks(
                            chunks=chunks,
                            embedding_model=embeddings,
                            url=QDRANT_URL,
                            api_key=qdrant_api_key,
                            collection_name=selected_collection_name # <-- MODIFIED
                        )
                    
                    except Exception as e:
                        st.error(f"An error occurred during upload: {e}")
                        st.exception(e)

        st.markdown("---")

if __name__ == "__main__":
    main()

