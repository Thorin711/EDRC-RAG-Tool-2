import streamlit as st
import os
import tempfile
import json
import openai
from llama_parse import LlamaParse

# --- Configuration ---
LLM_METADATA_MODEL = "gpt-5-mini" # Model for extracting metadata

# --- Helper Function: Extract Metadata with OpenAI ---

def extract_metadata_from_text(text_content: str) -> dict:
    """
    Uses OpenAI function calling to extract metadata from the first page text.
    """
    st.write("Extracting metadata using AI...")
    try:
        # Define the desired JSON structure for the LLM
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The main title of the paper"},
                "authors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of all author names"
                },
                "year": {"type": "string", "description": "The publication year (e.g., '2024')"},
                "doi": {"type": "string", "description": "The Digital Object Identifier (DOI)"}
            },
            "required": ["title", "authors", "year", "doi"]
        }
        
        # Note: We configure the client inside the function to use the secret
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=LLM_METADATA_MODEL,
            response_format={"type": "json_object", "schema": json_schema},
            messages=[
                {"role": "system", "content": "You are an expert academic librarian. Extract the title, authors, year, and DOI from the provided text. If a field is not present, return an empty string or list."},
                {"role": "user", "content": f"Here is the text from the first page of a research paper:\n\n---\n{text_content}\n---"}
            ]
        )
        
        metadata = json.loads(response.choices[0].message.content)
        st.success("AI metadata extraction complete.")
        return metadata

    except Exception as e:
        st.error(f"Error during AI metadata extraction: {e}")
        return {}

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="PDF to Markdown Converter", page_icon="📄", layout="wide")
    st.title("📄 PDF to Markdown Converter (with LlamaParse)")

    # --- Initialize Session State ---
    if "raw_markdown_body" not in st.session_state:
        st.session_state.raw_markdown_body = None
    if "extracted_metadata" not in st.session_state:
        st.session_state.extracted_metadata = None
    if "final_markdown" not in st.session_state:
        st.session_state.final_markdown = None
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None

    # --- API Key Checks ---
    openai_key = st.secrets.get("OPENAI_API_KEY")
    llama_key = st.secrets.get("LLAMA_CLOUD_API_KEY")

    if not openai_key or not llama_key:
        st.error("Missing API keys in Streamlit secrets!")
        if not openai_key:
            st.warning("`OPENAI_API_KEY` is not set.")
        if not llama_key:
            st.warning("`LLAMA_CLOUD_API_KEY` is not set. Get one from cloud.llamaindex.ai")
        st.stop()

    # --- 1. PDF Upload ---
    uploaded_file = st.file_uploader("Upload a PDF to process", type="pdf")

    if uploaded_file and not st.session_state.final_markdown:
        if st.session_state.raw_markdown_body is None: # Only process if not already done
            st.session_state.original_filename = os.path.splitext(uploaded_file.name)[0]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # --- Step 1: Parse with LlamaParse ---
                with st.spinner("Processing PDF with LlamaParse... This may take a moment."):
                    # Set the API key for LlamaParse
                    os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
                    
                    # LlamaParse returns a list of 'Document' objects
                    documents = LlamaParse(result_type="markdown").load_data(tmp_file_path)
                    
                    # Combine the text from all documents
                    st.session_state.raw_markdown_body = "\n\n".join([doc.text for doc in documents])
                    st.success("LlamaParse processing complete.")

                # --- Step 2: Extract Metadata ---
                if documents:
                    # Use text from the first document (usually contains the header)
                    first_page_text = documents[0].text
                    st.session_state.extracted_metadata = extract_metadata_from_text(first_page_text)
                else:
                    st.warning("LlamaParse returned no content.")
                    st.session_state.extracted_metadata = {}

            except Exception as e:
                st.error(f"An error occurred during parsing: {e}")
            finally:
                os.remove(tmp_file_path) # Clean up temp file
    
    # --- 2. Metadata Validation Form ---
    if st.session_state.extracted_metadata is not None and not st.session_state.final_markdown:
        st.header("2. Validate Extracted Metadata")
        st.info("Please review and correct the metadata extracted by the AI. This will be added as YAML front matter.")
        
        meta = st.session_state.extracted_metadata

        with st.form("metadata_form"):
            # Get default values, handling potential None or missing keys
            default_title = meta.get("title", "") or ""
            default_authors_list = meta.get("authors", []) or []
            default_authors_str = "\n".join(default_authors_list)
            default_year = meta.get("year", "") or ""
            default_doi = meta.get("doi", "") or ""

            # Form fields
            title = st.text_input("Title", value=default_title)
            authors_str = st.text_area("Authors (one per line)", value=default_authors_str, height=100)
            year = st.text_input("Year", value=default_year)
            doi = st.text_input("DOI", value=default_doi)
            
            submitted = st.form_submit_button("Generate Final Markdown")

            if submitted:
                # Process the form data
                authors_list = [a.strip() for a in authors_str.split('\n') if a.strip()]

                # Create YAML Front Matter
                yaml_front_matter = "---\n"
                yaml_front_matter += f'title: "{title.replace("\"", "\\\"")}"\n'
                yaml_front_matter += "authors:\n"
                if authors_list:
                    for author in authors_list:
                        yaml_front_matter += f'  - "{author.replace("\"", "\\\"")}"\n'
                else:
                    yaml_front_matter += "  - Not Available\n"
                yaml_front_matter += f'year: "{year}"\n'
                yaml_front_matter += f'doi: "{doi}"\n'
                yaml_front_matter += "---\n\n"

                # Combine with the markdown body
                st.session_state.final_markdown = yaml_front_matter + st.session_state.raw_markdown_body
                st.success("Final Markdown file generated!")

    # --- 3. Download Final File ---
    if st.session_state.final_markdown:
        st.header("3. Download Your File")
        
        md_filename = f"{st.session_state.original_filename}.md"
        
        st.download_button(
            label="Download .md File",
            data=st.session_state.final_markdown,
            file_name=md_filename,
            mime="text/markdown"
        )

        st.markdown("---")
        st.subheader("Final Markdown Preview")
        st.text_area("Markdown Content", st.session_state.final_markdown, height=500)

        if st.button("Process Another File"):
            # Clear all session state to reset the app
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
