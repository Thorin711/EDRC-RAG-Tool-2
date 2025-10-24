import streamlit as st
import os
import tempfile
import json
import openai
from llama_parse import LlamaParse

# --- Configuration ---
LLM_METADATA_MODEL = "gpt-4o-mini" # Model for extracting metadata

# --- Helper Function: Extract Metadata with OpenAI (MODIFIED) ---

def extract_metadata_from_text(text_content: str) -> dict:
    """
    Uses OpenAI to extract metadata from the first page text.
    """
    st.write("Extracting metadata using AI...")
    try:
        # --- MODIFIED PART ---
        # Instead of a schema, we put the instructions directly in the prompt
        # to guide the JSON output.
        prompt_schema_instructions = """
        You must return a single JSON object with the following exact keys:
        {
            "title": "The main title of the paper",
            "authors": ["List of all author names"],
            "year": "The publication year (e.g., '2024')",
            "doi": "The Digital Object Identifier (DOI)"
        }
        If a field is not present, return an empty string (for title, year, doi) 
        or an empty list (for authors).
        """
        
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=LLM_METADATA_MODEL,
            # This is the corrected part: We only specify the type, not the schema.
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": f"You are an expert academic librarian. Extract the metadata from the provided text. {prompt_schema_instructions}"},
                {"role": "user", "content": f"Here is the text from the first page of a research paper:\n\n---\n{text_content}\n---"}
            ]
        )
        
        metadata = json.loads(response.choices[0].message.content)
        st.success("AI metadata extraction complete.")
        return metadata

    except Exception as e:
        st.error(f"Error during AI metadata extraction: {e}")
        # Add a fallback to handle JSON parsing errors or other issues
        st.warning("AI extraction failed. Please fill in metadata manually.")
        # Return an empty structure so the form doesn't crash
        return {"title": "", "authors": [], "year": "", "doi": ""} 
    # --- END MODIFIED PART ---

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
                    
                    documents = LlamaParse(result_type="markdown").load_data(tmp_file_path)
                    
                    st.session_state.raw_markdown_body = "\n\n".join([doc.text for doc in documents])
                    st.success("LlamaParse processing complete.")

                # --- Step 2: Extract Metadata ---
                if documents:
                    first_page_text = documents[0].text
                    st.session_state.extracted_metadata = extract_metadata_from_text(first_page_text)
                else:
                    st.warning("LlamaParse returned no content.")
                    st.session_state.extracted_metadata = {"title": "", "authors": [], "year": "", "doi": ""}

            except Exception as e:
                st.error(f"An error occurred during parsing: {e}")
            finally:
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
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
                st.rerun() # Rerun to show the download section

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
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

