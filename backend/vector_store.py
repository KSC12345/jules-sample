import os # For file system operations like path joining and checking existence.
import chromadb # The ChromaDB client library for vector storage and retrieval.
# Import functions from document_processor.py for loading, chunking, and embedding text.
from backend.document_processor import load_txt, load_pdf, load_docx, chunk_text, generate_embeddings

# --- ChromaDB Client and Collection Initialization ---
# Attempt to initialize the ChromaDB client and get or create a collection.
# This setup is done once when the module is imported.
try:
    # Using PersistentClient to store ChromaDB data on disk, allowing persistence across runs.
    # The CHROMA_DB_PATH environment variable can be used to configure the storage path.
    # If not set, it defaults to a 'chroma_data' directory in the current working directory.
    # Inside Docker, os.getcwd() is typically '/app', so data is stored in '/app/chroma_data/'.
    chroma_data_dir = os.getenv("CHROMA_DB_PATH", "chroma_data")
    chroma_db_path = os.path.join(os.getcwd(), chroma_data_dir) 
    
    # Ensure the directory for persistent storage exists.
    if not os.path.exists(chroma_db_path):
        os.makedirs(chroma_db_path, exist_ok=True)
    
    # Initialize the PersistentClient with the specified path.
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Get or create a collection named "rag_collection".
    # A collection in ChromaDB is similar to a table in a relational database.
    # If using a custom sentence transformer model for embeddings with ChromaDB,
    # you might need to specify embedding_function when creating the collection.
    # However, here we are generating embeddings *before* adding to Chroma,
    # so we add raw documents along with their pre-computed embeddings.
    collection = client.get_or_create_collection(name="rag_collection")
    
    print(f"ChromaDB client initialized with persistent storage at: {chroma_db_path}")
    print(f"Collection '{collection.name}' loaded/created. Number of items: {collection.count()}")

except Exception as e:
    # If ChromaDB initialization fails, print an error and set client/collection to None.
    # This allows the application to start but indicates vector store functionality will be unavailable.
    print(f"Error initializing ChromaDB: {e}")
    client = None
    collection = None

# --- Core Vector Store Functions ---

def add_embeddings_to_collection(chunk_ids: list[str], text_chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """
    Adds text chunks, their pre-computed embeddings, and associated metadata to the ChromaDB collection.

    Args:
        chunk_ids (list[str]): A list of unique IDs for each text chunk.
        text_chunks (list[str]): The actual text content of the chunks.
        embeddings (list[list[float]]): A list of numerical embeddings corresponding to each text chunk.
        metadatas (list[dict]): A list of dictionaries containing metadata for each chunk (e.g., source document).

    Returns:
        bool: True if the addition was successful, False otherwise.
    """
    if collection is None:
        print("Error: ChromaDB collection is not initialized.")
        return False
    
    # Validate that all input lists have the same length, as they correspond element-wise.
    if not (len(chunk_ids) == len(text_chunks) == len(embeddings) == len(metadatas)):
        print("Error: Mismatch in lengths of IDs, chunks, embeddings, or metadatas.")
        return False
        
    try:
        # Add items to the collection. ChromaDB stores documents, their embeddings, metadatas, and IDs.
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=text_chunks, # Storing the original text chunk is useful for retrieval context.
            metadatas=metadatas
        )
        print(f"Successfully added {len(chunk_ids)} embeddings to the collection.")
        return True
    except Exception as e:
        print(f"Error adding embeddings to ChromaDB: {e}")
        return False

def process_and_embed_documents(documents_dir: str = "backend/documents_for_rag"):
    """
    Scans a specified directory for documents (.txt, .pdf, .docx), processes each document
    by loading its content, chunking the text, generating embeddings for these chunks,
    and finally adding them to the ChromaDB collection.

    Args:
        documents_dir (str): The path to the directory containing documents to be processed.
                             Defaults to "backend/documents_for_rag".

    Returns:
        dict: A status dictionary containing information about the processing, including
              number of files processed, chunks added, and total items in the collection.
              Returns an error status if critical components (ChromaDB, embedding model)
              are not initialized or if the directory is not found.
    """
    # Check if ChromaDB collection and the embedding generation function (and thus the model) are available.
    if collection is None or generate_embeddings is None: 
        print("Error: ChromaDB collection or SentenceTransformer model is not initialized.")
        return {"status": "error", "message": "ChromaDB or embedding model not initialized."}

    processed_files = 0
    total_chunks_added = 0
    
    # Ensure the documents_dir path is absolute or correctly resolved (e.g., within Docker).
    if not os.path.isabs(documents_dir):
        # Assuming this script runs in a context where /app is the root (like in Docker).
        documents_dir = os.path.join("/app", documents_dir)

    print(f"Scanning directory for documents: {documents_dir}")
    if not os.path.isdir(documents_dir):
        print(f"Error: Directory '{documents_dir}' not found.")
        return {"status": "error", "message": f"Directory '{documents_dir}' not found."}

    # Iterate over each file in the specified directory.
    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)
        text_content = ""
        file_extension = os.path.splitext(filename)[1].lower() # Get file extension, e.g., '.txt'

        print(f"Processing file: {filename} (type: {file_extension})")

        # Load content based on file type.
        if file_extension == '.txt':
            text_content = load_txt(file_path)
        elif file_extension == '.pdf':
            text_content = load_pdf(file_path)
        elif file_extension == '.docx':
            text_content = load_docx(file_path)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue # Move to the next file

        if not text_content: # If no content was extracted, skip this file.
            print(f"No content extracted from {filename}. Skipping.")
            continue

        # Chunk the extracted text.
        # Using smaller chunks (e.g., 100 words) can sometimes yield more fine-grained search results.
        text_chunks = chunk_text(text_content, chunk_size=100, overlap=10) 
        if not text_chunks:
            print(f"No chunks generated for {filename}. Skipping.")
            continue
        print(f"Generated {len(text_chunks)} chunks for {filename}.")

        # Generate embeddings for the text chunks.
        embeddings = generate_embeddings(text_chunks)
        if not embeddings or len(embeddings) != len(text_chunks): # Validate embedding generation
            print(f"Error generating embeddings for {filename} or mismatch in count. Skipping.")
            continue
        print(f"Generated {len(embeddings)} embeddings for {filename}.")

        # Prepare IDs and metadata for each chunk.
        # Chunk IDs should be unique.
        chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(text_chunks))]
        metadatas = [
            {"source_file": filename, "chunk_number": i, "original_extension": file_extension}
            for i in range(len(text_chunks))
        ]

        # Add the processed chunks and their embeddings to the ChromaDB collection.
        if add_embeddings_to_collection(chunk_ids, text_chunks, embeddings, metadatas):
            processed_files += 1
            total_chunks_added += len(text_chunks)
            print(f"Successfully processed and added {len(text_chunks)} chunks from {filename}.")
        else:
            print(f"Failed to add chunks from {filename} to collection.")

    # Return a summary of the processing.
    return {
        "status": "success",
        "processed_files": processed_files,
        "total_chunks_added": total_chunks_added,
        "collection_total_items": collection.count() if collection else 0
    }

# --- Deprecated/Alternative Functions (kept for reference or specific use cases) ---

def add_raw_document_to_collection(doc_id: str, document: str, metadata: dict):
    """
    Adds a single raw document string to the ChromaDB collection.
    This is different from `add_embeddings_to_collection` as it relies on ChromaDB's
    internal or configured embedding function to process the document.
    This function is less used in this RAG setup where embeddings are pre-computed.

    Args:
        doc_id (str): Unique ID for the document.
        document (str): The raw text content of the document.
        metadata (dict): Metadata associated with the document.

    Returns:
        bool: True if successful, False otherwise.
    """
    if collection is None:
        print("Error: ChromaDB collection is not initialized.")
        return False
    try:
        # Note: If the collection was created *without* a specific embedding function,
        # and you add documents directly without providing embeddings, ChromaDB might use
        # a default embedding function (e.g., Sentence Transformers all-MiniLM-L6-v2 in some versions)
        # or it might error if it expects explicit embeddings.
        # For this project, we primarily use `add_embeddings_to_collection`.
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[doc_id]
        )
        print(f"Raw document with ID '{doc_id}' added successfully. ChromaDB will handle its embedding if configured.")
        return True
    except Exception as e:
        print(f"Error adding raw document to ChromaDB: {e}")
        return False

def query_collection_with_text(query_text: str, n_results: int = 1):
    """
    Queries the ChromaDB collection using a raw query text.
    This function relies on ChromaDB's internal or configured embedding function
    to convert the query text into an embedding before performing the search.

    Args:
        query_text (str): The text to query the collection with.
        n_results (int): The number of results to retrieve.

    Returns:
        dict or None: The query results from ChromaDB, or None if an error occurs.
    """
    if collection is None:
        print("Error: ChromaDB collection is not initialized.")
        return None
    try:
        results = collection.query(
            query_texts=[query_text], # ChromaDB handles embedding this text
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Specify what data to include in results
        )
        print(f"Text query successful: {results}")
        return results
    except Exception as e:
        print(f"Error querying ChromaDB with text: {e}")
        return None

def query_collection_with_embedding(query_embedding: list[float], n_results: int = 1):
    """
    Queries the ChromaDB collection using a pre-computed query embedding.
    This is the primary query method used in this RAG application, as query embeddings
    are generated by the same model used for document embeddings.

    Args:
        query_embedding (list[float]): The numerical embedding of the query.
        n_results (int): The number of results to retrieve.

    Returns:
        dict or None: The query results from ChromaDB, or None if an error occurs or input is invalid.
    """
    if collection is None:
        print("Error: ChromaDB collection is not initialized.")
        return None
    if not query_embedding: # Validate that the query embedding is not empty.
        print("Error: Query embedding is empty.")
        return None
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding], # Pass the pre-computed embedding
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'] # Include document text, metadata, and distance scores
        )
        print(f"Embedding query successful: {results}")
        return results
    except Exception as e:
        print(f"Error querying ChromaDB with embedding: {e}")
        return None

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    # This block executes only when the script is run directly (e.g., `python vector_store.py`).
    # It's useful for testing the functionalities of this module independently.
    if collection: # Check if ChromaDB collection was initialized
        print("\nTesting ChromaDB vector store functions...")
        
        # --- Test Document Processing and Embedding ---
        print("\n--- Testing Document Processing and Embedding ---")
        # Determine the path to the documents directory.
        # This handles running the script from different locations (e.g., /app or /app/backend).
        current_script_dir = os.path.dirname(__file__)
        docs_path_relative_to_script = 'documents_for_rag'
        # Try path relative to script first (e.g. if running from backend/)
        docs_path = os.path.join(current_script_dir, docs_path_relative_to_script)
        
        if not os.path.exists(docs_path): # Fallback if running from /app (then backend/documents_for_rag)
             docs_path_relative_to_app = os.path.join("backend", docs_path_relative_to_script)
             if os.path.exists(os.path.join(os.getcwd(), docs_path_relative_to_app)):
                 docs_path = docs_path_relative_to_app # Use path relative to /app
             else: # If still not found, create a dummy one in default location for test
                 os.makedirs(os.path.join(current_script_dir, docs_path_relative_to_script), exist_ok=True)
                 with open(os.path.join(current_script_dir, docs_path_relative_to_script, "dummy.txt"), "w") as f:
                     f.write("This is a dummy file for vector_store testing.")
                 print(f"Created dummy document for testing at {docs_path}")


        processing_result = process_and_embed_documents(documents_dir=docs_path)
        print(f"Document processing result: {processing_result}")
        
        if collection: # Re-check collection as it might be reset by tests or errors
            print(f"Total items in collection after processing: {collection.count()}")

            # --- Test Querying (if documents were added) ---
            if collection.count() > 0:
                print("\n--- Testing Query with Text (Relies on ChromaDB's default embedding if not configured) ---")
                # This query might not be ideal if Chroma's default embedder differs from our SentenceTransformer
                query_text_results = query_collection_with_text(
                    query_text="What is a fox?", # Example query
                    n_results=1
                )
                if query_text_results and query_text_results.get('documents') and query_text_results['documents'][0]:
                    print(f"Text query results for 'What is a fox?': {query_text_results['documents'][0][0]}")
                else:
                    print("Text query for 'What is a fox?' failed or returned no relevant results.")

                print("\n--- Testing Query with Pre-computed Embedding ---")
                # Generate an embedding for a test query using our application's embedding function.
                if generate_embeddings and model: # Check if our embedding model is loaded
                    test_query_embedding_list = generate_embeddings(["renewable energy"])
                    if test_query_embedding_list:
                        test_query_embedding = test_query_embedding_list[0]
                        embedding_query_results = query_collection_with_embedding(
                            query_embedding=test_query_embedding,
                            n_results=1
                        )
                        if embedding_query_results and embedding_query_results.get('documents') and embedding_query_results['documents'][0]:
                            print(f"Embedding query results for 'renewable energy': {embedding_query_results['documents'][0][0]}")
                        else:
                            print("Embedding query for 'renewable energy' failed or returned no relevant results.")
                    else:
                        print("Could not generate test embedding for 'renewable energy'.")
                else:
                    print("SentenceTransformer model not loaded, skipping specific embedding query test.")
            else:
                print("\nSkipping query tests as collection is empty after processing attempt.")
    else:
        print("Cannot run ChromaDB tests as collection is not initialized.")
