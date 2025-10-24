# -*- coding: utf-8 -*-
"""
Streamlit App to process PDFs using a personal GROBID server,
review and EDIT the extracted YAML/text, and download the final Markdown.
"""

import streamlit as st
import requests
import re
import os
import time
from bs4 import BeautifulSoup
import unicodedata

# --- CONFIGURATION ---

# This is your personal GROBID server URL.
GROBID_API_URL = "https://thorin711-edrc-grobid.hf.space/api/processFulltextDocument"

REQUEST_TIMEOUT = 180  # 3 minutes for slow server wake-up and processing
MAX_RETRIES = 2
RETRY_DELAY = 5      # 5 seconds to wait between retries

# --- 1. GROBID API CALL LOGIC ---

def sanitize_filename(filename):
    """
    Cleans a filename by removing special characters, normalizing,
    and ensuring a basic name if all chars are stripped.
    """
    try:
        # Normalize unicode characters
        normalized_name = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
        # Keep only alphanumeric, dots, and hyphens. Replace spaces.
        safe_name = re.sub(r'[^\w.-]', '', normalized_name.replace(' ', '-'))
        # Ensure it's not empty
        if not safe_name:
            return "document.pdf"
        return safe_name
    except Exception as e:
        print(f"Error sanitizing filename: {e}")
        return "document.pdf"

# NOTE: Removed @st.cache_data to prevent file stream errors
def call_grobid_api(pdf_bytes, filename):
    """
    Calls the configured GROBID API to process a PDF.
    Includes retries and a longer timeout.
    """
    if GROBID_API_URL.startswith("https://YOUR-HF-USERNAME"):
        st.error("Please update the `GROBID_API_URL` in the grobid_processor_app.py file with your Hugging Face URL.", icon="🚨")
        return None, "Configuration Error"

    if len(pdf_bytes) == 0:
        return None, "Skipped processing 0-byte file."
    
    safe_filename = sanitize_filename(filename)

    # Build the full multipart/form-data payload
    multipart_payload = {
        'input': (safe_filename, pdf_bytes, 'application/pdf'),
        'consolidateHeader': (None, '1'),
        'consolidateCitations': (None, '0'),
        'includeRawCitations': (None, '0'),
        'includeRawAffiliations': (None, '0'),
        'teiCoordinates': (None, '0')
    }
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            st.write(f"Connecting to your GROBID server (Attempt {attempt + 1}/{MAX_RETRIES + 1})...")
            response = requests.post(
                GROBID_API_URL,
                files=multipart_payload, 
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                return response.text, None
            else:
                error_message = f"GROBID API returned an error (Status {response.status_code}):"
                error_detail = response.text[:500] 
                
                if "Cannot invoke \"java.io.InputStream.close()\"" in error_detail:
                    error_message = (
                        "GROBID server error: 'InputStream is null'. "
                        "This often means the PDF file is corrupt, encrypted, "
                        "or password-protected. Please try a different PDF."
                    )
                else:
                     error_message += f"\nError detail: {error_detail}"
                
                if attempt == MAX_RETRIES:
                    return None, error_message
                
        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                return None, f"Request timed out after {REQUEST_TIMEOUT} seconds. Your server might be slow or the PDF is too complex."
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                return None, f"A connection error occurred: {e}"
        
        st.write(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    
    return None, "Unknown error after all retries."

# --- 2. XML TO MARKDOWN LOGIC ---

def parse_xml_to_markdown(xml_content, filename_for_download):
    """
    Parses GROBID's XML, extracts YAML and body, and returns them.
    Returns (yaml_str, body_str, full_md_str, download_filename)
    """
    try:
        soup = BeautifulSoup(xml_content, 'lxml-xml')

        # --- Pre-process to remove unwanted sections ---
        unwanted_section_headings = [
            "references", "bibliography", "acknowledgement", "acknowledgements",
            "declaration of competing interest", "credit authorship contribution statement"
        ]
        for div in soup.find_all('div'):
            head = div.find('head')
            if head and head.get_text(strip=True).lower() in unwanted_section_headings:
                div.decompose()

        # --- 1. Extract Metadata (with safety checks) ---
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

        title = title.replace('"', '\\"')

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

        # --- 2. Create YAML Front Matter ---
        yaml_front_matter = "---\n"
        yaml_front_matter += f'title: "{title}"\n'
        yaml_front_matter += "authors:\n"
        if authors:
            for author in authors:
                yaml_front_matter += f'  - "{author}"\n'
        else:
            yaml_front_matter += "  - Not Available\n"
        yaml_front_matter += f'doi: "{doi}"\n'
        yaml_front_matter += f'year: "{year}"\n'
        yaml_front_matter += "---\n\n" # IMPORTANT: Keep the newlines

        # --- 3. Process the Body Content ---
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

        # --- 4. Assemble and return ---
        final_markdown = yaml_front_matter + markdown_body
        
        base_name = os.path.splitext(filename_for_download)[0]
        download_filename = f"{base_name}.md"
        
        # Return the individual parts for editing
        return yaml_front_matter, markdown_body, download_filename

    except Exception as e:
        st.exception(e) # Print the full error to Streamlit
        return None, None, f"Error parsing XML: {e}"

# --- 3. STREAMLIT APP UI ---

def main():
    st.set_page_config(layout="wide", page_title="GROBID PDF Processor")
    st.title("📄 GROBID PDF to Markdown Converter")
    st.markdown("This app uses your personal GROBID server on Hugging Face to extract metadata and text from academic papers.")

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
        # Use file.id as a unique key for widgets
        unique_key = uploaded_file.id
        st.header(f"Processing: `{uploaded_file.name}`")
        
        with st.spinner(f"Contacting your GROBID server... This can take 1-3 minutes if the server is waking up."):
            pdf_bytes = uploaded_file.getvalue()
            xml_result, error_msg = call_grobid_api(pdf_bytes, uploaded_file.name)

        if error_msg:
            st.error(f"**Failed to process `{uploaded_file.name}`:**\n\n{error_msg}")
            continue

        st.success(f"Successfully processed `{uploaded_file.name}`!")

        with st.spinner("Parsing XML and building Markdown..."):
            yaml_data, md_body, dl_filename = parse_xml_to_markdown(xml_result, uploaded_file.name)
        
        if not yaml_data:
            st.error(f"Failed to parse XML for `{uploaded_file.name}`: {dl_filename}")
            continue

        # --- Display Results ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Metadata (YAML Review & Edit)")
            
            # <-- MODIFICATION 1: Changed st.code to st.text_area -->
            # This makes the YAML editable.
            edited_yaml = st.text_area(
                "Edit YAML Metadata:",
                value=yaml_data,
                height=300,
                key=f"yaml_{unique_key}" # Unique key per file
            )
        
        with col2:
            st.subheader("Full Markdown Content (Review & Edit)")
            
            # <-- MODIFICATION 2: Captured the output of the text_area -->
            # This makes the body text editable.
            edited_body = st.text_area(
                "Review the full extracted text:",
                value=md_body,
                height=500,
                key=f"body_{unique_key}" # Unique key per file
            )

        # <-- MODIFICATION 3: Re-create the download button in col1 -->
        # This ensures it uses the *edited* text from both boxes.
        with col1:
            st.write("") # Spacer
            
            # Combine the *edited* parts for the final download
            final_markdown_for_download = edited_yaml + edited_body
            
            st.download_button(
                label=f"Download `{dl_filename}`",
                data=final_markdown_for_download, # Use the edited content
                file_name=dl_filename,
                mime="text/markdown",
                key=f"dl_{unique_key}" # Unique key per file
            )

        st.markdown("---")

if __name__ == "__main__":
    main()

