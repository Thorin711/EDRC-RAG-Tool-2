# EDRC Research Paper RAG Tool

This repository contains a Retrieval-Augmented Generation (RAG) tool designed to search and synthesize information from a collection of research papers. The tool consists of a data processing pipeline to create a vector database from PDF and XML files, and a Streamlit web application for user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Running the Application](#2-running-the-application)
- [File Descriptions](#file-descriptions)

## Project Overview

The primary goal of this project is to provide an intelligent search interface for a corpus of research documents. Instead of simple keyword matching, it uses semantic search to find the most relevant document passages and an LLM to generate a summarized answer based on the retrieved information, complete with citations.

**Key Features:**
- **End-to-End Pipeline:** Scripts to process raw PDFs into a structured vector database.
- **Semantic Search:** Utilizes sentence embeddings to find conceptually related text.
- **AI-Powered Summarization:** Generates a concise, cited summary of the findings.
- **Interactive UI:** A user-friendly Streamlit web app to ask questions and explore results.
- **Modular and Extensible:** The data pipeline and app are separated for easier maintenance and extension.

## Data Processing Pipeline

The data processing pipeline converts raw research papers (ideally in PDF format) into a Chroma vector database. This is a multi-step process:

1.  **PDF to TEI XML (`simple_process_pdfs.py`):**
    -   Uses the [GROBID](https://github.com/kermitt2/grobid) service to parse PDFs.
    -   GROBID extracts structured text and metadata (title, authors, abstract, sections) and outputs it as a TEI XML file. This is crucial for preserving the document's structure.

2.  **TEI XML to Markdown (`xml_to_md.py`):**
    -   The script converts the TEI XML files into clean Markdown files.
    -   It extracts metadata (title, authors, year, DOI) and places it into a YAML front matter block.
    -   The main content is converted to Markdown, with section headings preserved. Unnecessary sections like "References" and "Acknowledgements" are removed.

3.  **Markdown to Vector Database (`create_vectordb.py`):**
    -   Loads the Markdown files, including their front matter metadata.
    -   Splits the documents into smaller chunks based on their Markdown headers. This keeps related paragraphs together under their original section headings.
    -   Uses a Hugging Face embedding model (`BAAI/bge-large-en-v1.5`) to generate a vector representation for each text chunk.
    -   Stores these chunks and their embeddings in a persistent Chroma vector database.

## Setup and Installation

### Prerequisites

-   **Python 3.8+**
-   **GROBID:** You must have a running GROBID instance. The easiest way to set one up is using Docker:
    ```bash
    docker pull lfoppiano/grobid:0.8.0
    docker run -t --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
    ```
    The GROBID service will be available at `http://localhost:8070`.

-   **OpenAI API Key:** For the AI-enhanced search and summarization features, you need an OpenAI API key.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Streamlit secrets for your OpenAI API Key:**
    -   Create a file at `~/.streamlit/secrets.toml` (or inside your project directory at `.streamlit/secrets.toml`).
    -   Add your key to this file:
        ```toml
        OPENAI_API_KEY="sk-..."
        ```

## Usage

### 1. Data Preparation

To create your own vector database, follow these steps. Note that the paths in the scripts are hardcoded and will need to be adjusted to your local file structure.

1.  **Place PDFs:** Put all your research paper PDFs into a single directory (e.g., `C:/data/pdfs`).

2.  **Run GROBID Processing (`simple_process_pdfs.py`):**
    -   Open `simple_process_pdfs.py`.
    -   Set `in_path` to your PDF directory and `out_path` to where you want the XML files to be saved.
    -   Ensure your GROBID Docker container is running.
    -   Run the script: `python simple_process_pdfs.py`

3.  **Run XML to Markdown Conversion (`xml_to_md.py`):**
    -   Open `xml_to_md.py`.
    -   Set `input_dir` to the directory containing the XML files from the previous step.
    -   Set `output_dir` to where you want the final Markdown files to be saved.
    -   Run the script: `python xml_to_md.py`

4.  **Create the Vector Database (`create_vectordb.py`):**
    -   Open `create_vectordb.py`.
    -   Set `md_output_dir` to the directory containing your Markdown files.
    -   Set `db_persist_dir` to the name of the folder where the vector database will be created (e.g., `./my_vector_db`).
    -   Run the script: `python create_vectordb.py`. This may take some time depending on the number of documents.

### 2. Running the Application

Once you have your vector database(s), you can run the Streamlit app. The app is configured to download pre-built databases from GitHub Releases on its first run, so you can try it out without preparing your own data.

1.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Using the App:**
    -   The app will automatically download and set up the default vector databases if they are not found locally.
    -   Select which database you want to search ("Full Database" or "Journal Articles Only").
    -   Enter your research question in the text box.
    -   Adjust the number of results to retrieve using the slider.
    -   Toggle "AI-Enhanced Search" to have an LLM refine your query for better results.
    -   Toggle "Generate AI Summary" to get a synthesized answer with citations.
    -   Click "Search". The results will be displayed below.

## File Descriptions

-   `app.py`: The main Streamlit web application.
-   `create_vectordb.py`: Script to create a Chroma vector database from processed Markdown files.
-   `simple_process_pdfs.py`: Script to process PDFs with a GROBID service to generate TEI XML.
-   `xml_to_md.py`: Script to convert TEI XML files into Markdown with YAML front matter.
-   `requirements.txt`: A list of all the Python packages required to run the project.
-   `README.md`: This file.