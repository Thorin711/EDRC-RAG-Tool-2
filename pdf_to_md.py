# -*- coding: utf-8 -*-
"""
This Streamlit application allows users to upload PDF files, process them
using the public GROBID API, review the extracted metadata, and download
the final structured Markdown files.

It combines PDF processing and XML-to-Markdown conversion into one
interactive workflow.
"""

import streamlit as st
import requests
import os
import re
from bs4 import BeautifulSoup
import time

# --- CONFIGURATION ---
# Use the public Hugging Face GROBID API endpoint
GROBID_API_URL = "https://kermitt2-grobid.hf.space/api/processFulltextDocument"

# --- API FUNCTION ---

def call_grobid_api(pdf_bytes):
    """
    Calls the GROBID API with the bytes of a single PDF file.

    Args:
        pdf_bytes (bytes): The content of the PDF file.

    Returns:
        str: The resulting TEI XML as a string, or None on failure.
    """
    try:
        # The 'files' parameter takes a dict-like object
        files = {'inputFile': pdf_bytes}
        
        # Add a timeout to handle potential API delays
        response = requests.post(GROBID_API_URL, files=files, timeout=60)

        if response.status_code == 200:
            return response.text
        else:
            st.error(f"GROBID API returned an error (Status {response.status_code}):")
            st.error(response.text)
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to GROBID API: {e}")
        return None

# --- PARSING FUNCTION (Adapted from your xml_to_md.py) ---

def parse_grobid_xml(xml_content):
    """
    Parses a GROBID TEI XML string into YAML front matter and Markdown.

    Args:
        xml_content (str): The TEI XML content from GROBID.

    Returns:
        tuple: (yaml_front_matter, final_markdown, base_filename)
               Returns (None, None, None) on parsing failure.
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
        else:
            body = soup.find('body')
            if body and body.find_all('div'):
                first_p = body.find_all('div')[0].find('p')
                if first_p:
                    title = first_p.get_text(strip=True)

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
        yaml_front_matter += "---"  # No newline needed for st.code

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

        # --- 4. Assemble Final Markdown ---
        final_markdown_body = "\n\n".join(markdown_body_parts)
        final_markdown = f"{yaml_front_matter}\n\n{final_markdown_body}"

        # --- 5. Determine Filename ---
        # Try to get filename from <title>
        base_name_from_title = re.sub(r'[^a-z0-9\s-]', '', title.lower()).strip()
        base_name_from_title = re.sub(r'[\s-]+', '-', base_name_from_title)
        base_name = base_name_from_title[:50] or "processed-paper" # Limit length

        return yaml_front_matter, final_markdown, base_name

    except Exception as e:
        st.error(f"An error occurred during XML parsing: {e}")
        st.expander("Show Raw XML that failed parsing:").code(xml_content, language='xml')
        return None, None, None

# --- STREAMLIT APP UI ---

def main():
    st.set_page_config(layout="wide", page_title="GROBID PDF Processor")
    st.title("📄 GROBID PDF-to-Markdown Processor")
    st.markdown("Upload your PDF files below. The app will process them using the public **Hugging Face GROBID API**, then show you the extracted metadata (YAML) for review before you download the final Markdown file.")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Uploaded PDFs"):
        if uploaded_files:
            st.info(f"Starting processing for {len(uploaded_files)} file(s)...")
            
            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Alternate columns for results
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f"---")
                    st.subheader(f"Processing: `{uploaded_file.name}`")
                    
                    with st.spinner(f"Calling GROBID API for {uploaded_file.name}... (This can take a minute)"):
                        pdf_bytes = uploaded_file.getvalue()
                        xml_result = call_grobid_api(pdf_bytes)
                    
                    if xml_result:
                        st.success(f"GROBID processing complete for {uploaded_file.name}.")
                        
                        with st.spinner(f"Parsing XML for {uploaded_file.name}..."):
                            yaml_data, md_data, base_name = parse_grobid_xml(xml_result)
                        
                        if yaml_data:
                            st.markdown("#### 1. Review Extracted YAML:")
                            st.code(yaml_data, language='yaml')
                            
                            st.markdown("#### 2. Review and Download Markdown:")
                            st.text_area(
                                "Full Markdown Content",
                                md_data,
                                height=300,
                                key=f"md_area_{uploaded_file.name}"
                            )
                            
                            st.download_button(
                                label=f"Download {base_name}.md",
                                data=md_data,
                                file_name=f"{base_name}.md",
                                mime="text/markdown",
                                key=f"dl_button_{uploaded_file.name}"
                            )
                    else:
                        st.error(f"Failed to process {uploaded_file.name} with GROBID.")
                        
            st.markdown("---")
            st.success("All processing complete!")
        else:
            st.warning("Please upload at least one PDF file before processing.")

if __name__ == "__main__":
    main()
