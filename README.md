# Research Paper RAG Tool

This repository contains a Retrieval-Augmented Generation (RAG) tool designed to search and synthesize information from a collection of research papers. The tool consists of a suite of Streamlit applications for data processing, metadata administration, and user-facing search.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Usage](#usage)
  - [1. Data Processing (`pdf_to_md.py`)](#1-data-processing-pdf_to_mdpy)
  - [2. Search and Query (`app.py`)](#2-search-and-query-apppy)
  - [3. Metadata Management (`admin.py`)](#3-metadata-management-adminpy)
- [File Descriptions](#file-descriptions)

## Project Overview

The primary goal of this project is to provide an intelligent search interface for a corpus of research documents. Instead of simple keyword matching, it uses semantic search to find the most relevant document passages and an LLM to generate a summarized answer based on the retrieved information, complete with citations.

**Key Features:**
- **Interactive Data Pipeline:** A Streamlit app (`pdf_to_md.py`) for processing PDFs, reviewing extracted text, and uploading to a vector database.
- **Semantic Search:** Utilizes sentence embeddings to find conceptually related text.
- **AI-Powered Summarization:** Generates a concise, cited summary of the findings.
- **Web-Based UI:** A user-friendly Streamlit web app (`app.py`) to ask questions and explore results.
- **Metadata Management:** An admin panel (`admin.py`) to find and batch-update document metadata.

## System Architecture

The system is built around three core Streamlit applications that interact with a Qdrant Cloud vector database.

1.  **`pdf_to_md.py` (PDF Uploader & Processor):**
    -   Users upload PDF documents through the Streamlit interface.
    -   The app calls a remote **GROBID** service to parse the PDF and extract structured XML.
    -   The XML is converted to Markdown, and metadata (title, authors, etc.) is extracted.
    -   The user can review and edit the extracted text and metadata directly in the app.
    -   Upon confirmation, the app splits the document into chunks and uploads them to a specified **Qdrant** collection.

2.  **`app.py` (Main Search Application):**
    -   The primary interface for end-users.
    -   Connects to the Qdrant database to perform semantic search.
    -   Allows users to select different collections (e.g., "Full Database", "Journal Articles Only").
    -   Features AI-powered query enhancement and result summarization using OpenAI models.
    -   Displays search results with metadata, content snippets, and links to the source.

3.  **`admin.py` (Metadata Admin Panel):**
    -   A utility for database maintenance.
    -   Allows an administrator to search for documents by title.
    -   Retrieves all chunks associated with a title and allows for batch updates of their shared metadata (e.g., correcting a typo in the title, adding a DOI).

## Setup and Installation

### Prerequisites

-   **Python 3.8+**
-   **GROBID:** The data processing app is configured to use a remote GROBID server. The default is a free Hugging Face Space (`https://thorin711-edrc-grobid.hf.space/api/processFulltextDocument`), but you can substitute your own.
-   **OpenAI API Key:** For the AI-enhanced search and summarization features.
-   **Qdrant API Key:** To connect to your Qdrant Cloud database.

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

4.  **Set up Streamlit secrets:**
    -   Create a file at `.streamlit/secrets.toml` inside your project directory.
    -   Add your API keys to this file:
        ```toml
        OPENAI_API_KEY="sk-..."
        QDRANT_API_KEY="your-qdrant-api-key"
        ```

## Usage

Each part of the system is a standalone Streamlit application.

### 1. Data Processing (`pdf_to_md.py`)

This app is for adding new documents to the vector database.

1.  **Launch the app:**
    ```bash
    streamlit run pdf_to_md.py
    ```

2.  **Using the App:**
    -   Select the target Qdrant collection (e.g., "Full Database").
    -   Upload one or more PDF files.
    -   The app will process each PDF using GROBID. This may take a few minutes.
    -   Review and edit the extracted metadata (title, authors, etc.) and the body text for each document.
    -   Click "Confirm & Upload to Vector DB" to chunk the document and add it to the selected collection.

### 2. Search and Query (`app.py`)

This is the main application for searching the document collections.

1.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```

2.  **Using the App:**
    -   Select which database collection you want to search.
    -   Enter your research question in the text box.
    -   Use the toggles to enable "AI-Enhanced Search" or "Generate AI Summary."
    -   Click "Search." The results will be displayed below.

### 3. Metadata Management (`admin.py`)

This app is for correcting or updating metadata for documents already in the database.

1.  **Launch the app:**
    ```bash
    streamlit run admin.py
    ```

2.  **Using the App:**
    -   Select the database collection you want to edit.
    -   Search for a document by its title.
    -   All document chunks matching that title will be retrieved.
    -   Enter the corrected metadata in the form.
    -   Click "Save Changes" to update the metadata for all selected chunks simultaneously.

## File Descriptions

-   `app.py`: The main user-facing Streamlit application for search and retrieval.
-   `pdf_to_md.py`: A Streamlit app for processing PDFs and uploading them to the Qdrant vector database.
-   `admin.py`: A Streamlit utility app for batch-editing document metadata in Qdrant.
-   `requirements.txt`: A list of the Python packages required to run the project.
-   `README.md`: This file.

## To Do

- Improve Database metadatas
- Fix Stability
- Add more files
- Double check files in EDRC only
