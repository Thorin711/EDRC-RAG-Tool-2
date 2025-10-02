import os
import frontmatter
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. DEFINE PATHS ---
md_output_dir = r'C:\Users\td00654\Documents\EDRC LLM Project\CREDS Papers\CREDS - Full'
db_persist_dir = r'./vector_db'

# --- Custom function to load documents with front matter ---
def load_documents_from_directory(directory):
    """
    Loads all markdown files from a directory, parsing their front matter
    and creating LangChain Document objects.
    """
    documents = []
    print("Loading documents with custom loader...")
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            post = frontmatter.load(filepath)
            
            if 'authors' in post.metadata and isinstance(post.metadata['authors'], list):
                post.metadata['authors'] = ", ".join(post.metadata['authors'])

            if 'year' in post.metadata and post.metadata['year'] is not None:
                post.metadata['year'] = str(post.metadata['year'])
            
            post.metadata['source'] = filepath
            doc = Document(page_content=post.content, metadata=post.metadata)
            documents.append(doc)
            
    return documents

print("Starting the vector database creation process...")

# --- 2. LOAD DOCUMENTS ---
documents = load_documents_from_directory(md_output_dir)
print(f"Successfully loaded {len(documents)} documents.")

# --- 3. SPLIT DOCUMENTS INTO CHUNKS (NEW HEADER-AWARE METHOD) ---
print("Splitting documents based on Markdown headers...")

# Define the headers we want to split on
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# Create the Markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    return_each_line=False # This keeps paragraphs together
)

# Process each document and merge metadata
all_chunks = []
for doc in documents:
    # Split the content of the document
    chunks = markdown_splitter.split_text(doc.page_content)
    
    # The splitter creates chunks with header metadata.
    # We now add the original document's metadata (title, authors, etc.) to each chunk.
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)
    
    # Add the processed chunks to our main list
    all_chunks.extend(chunks)

# Overwrite the 'docs' variable with our new, smarter chunks
docs = all_chunks
print(f"Split the documents into {len(docs)} header-aware chunks.")


# --- 4. CREATE EMBEDDINGS ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model loaded.")

# --- 5. STORE IN VECTOR DATABASE ---
print("Creating and persisting the vector database... (This may take a few minutes)")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=db_persist_dir
)
print("Vector database created successfully!")

#%% --- EXAMPLE SEARCH ---
print("\n--- Running an example search ---")
query = "What are the effects of policy on renewable energy adoption?"
results = vector_store.similarity_search(query, k=3)

print(f"Query: '{query}'")
print("\nTop 3 results:")
for i, doc in enumerate(results):
    title = doc.metadata.get('title', 'No Title Found')
    print(f"--- Result {i+1} ---")
    print(f"Source Title: {title}")
    # --- NEW: Display header metadata if it exists ---
    headers = [doc.metadata.get(f'Header {i}') for i in range(1, 4) if doc.metadata.get(f'Header {i}')]
    if headers:
        print(f"Section: {' > '.join(headers)}")
    print(f"Content: {doc.page_content[:300]}...\n")