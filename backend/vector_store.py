import os
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib # For generating unique IDs for documents

from . import config # Import config for paths, model names, etc.

# --- Sentence Transformer Model Initialization ---
try:
    # Initialize the SentenceTransformer model using the name from config
    # This model will be used to convert text (documents and queries) into numerical embeddings.
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    print(f"SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}': {e}")
    print("Embedding generation and querying will not be available.")
    embedding_model = None

# --- ChromaDB Client and Collection Initialization ---
try:
    # Use ChromaDB path from config. Ensure it's an absolute path or resolved correctly.
    # If CHROMA_DB_PATH is relative, it's relative to the CWD.
    # For robustness, especially in Docker, an absolute path in config is recommended.
    chroma_db_path = config.CHROMA_DB_PATH
    
    if not os.path.isabs(chroma_db_path):
        # If path is relative, make it relative to the /app directory (common CWD in Docker)
        # This is a heuristic. Ideally, config.CHROMA_DB_PATH should be set to an absolute path
        # or a path that's reliably relative to the project root.
        # For this project, assuming CWD is /app where backend/ is a subdirectory.
        # If CHROMA_DB_PATH is "./chroma_data", it becomes "/app/chroma_data".
        chroma_db_path = os.path.join(os.getcwd(), chroma_db_path)
        print(f"Resolved relative CHROMA_DB_PATH to: {chroma_db_path}")

    # Ensure the directory for persistent storage exists.
    if not os.path.exists(chroma_db_path):
        os.makedirs(chroma_db_path, exist_ok=True)
        print(f"Created ChromaDB directory at: {chroma_db_path}")
    
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Use collection name from config
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    
    print(f"ChromaDB client initialized. Storage path: {chroma_db_path}")
    print(f"Collection '{config.CHROMA_COLLECTION_NAME}' loaded/created. Items: {collection.count()}")

except Exception as e:
    print(f"Error initializing ChromaDB with path '{config.CHROMA_DB_PATH}' and collection '{config.CHROMA_COLLECTION_NAME}': {e}")
    client = None
    collection = None
    if embedding_model is None: # if sentence transformer also failed
        print("Both SentenceTransformer model and ChromaDB failed to initialize. Core functionality disabled.")

# --- Core Vector Store Functions ---

def add_documents(documents: list[dict]):
    """
    Adds a list of documents (React components) to the ChromaDB collection.
    Each document is a dictionary with 'content' and 'metadata' (e.g., filename).
    Embeddings are generated for the 'content' of each document.

    Args:
        documents (list[dict]): A list of document objects.
                                Each object must have a 'content' (str) key and
                                a 'metadata' (dict) key.

    Returns:
        bool: True if all documents were added successfully, False otherwise.
    """
    if collection is None or embedding_model is None:
        print("Error: ChromaDB collection or SentenceTransformer model is not initialized. Cannot add documents.")
        return False
    
    if not documents:
        print("No documents provided to add.")
        return True # Technically successful, as there was nothing to fail on.

    doc_contents = []
    doc_metadatas = []
    doc_ids = []

    for i, doc in enumerate(documents):
        if "content" not in doc or "metadata" not in doc or "filename" not in doc["metadata"]:
            print(f"Warning: Document at index {i} is missing 'content' or 'metadata.filename'. Skipping.")
            continue
        
        doc_contents.append(doc["content"])
        # Ensure metadata is a flat dictionary of simple types for ChromaDB
        metadata = {k: str(v) for k, v in doc["metadata"].items()}
        doc_metadatas.append(metadata)
        
        # Create a unique ID for the document, e.g., based on its content hash or filename
        # Using filename as ID, assuming filenames are unique within the `retrieved_components` dir.
        # If not unique, consider hashing content or using a UUID.
        # For now, prefixing with collection name for more uniqueness if multiple collections were used.
        doc_id = f"{config.CHROMA_COLLECTION_NAME}_{doc['metadata']['filename']}"
        # Alternative: Use a hash of content for more robustness if filenames aren't strictly unique
        # content_hash = hashlib.md5(doc["content"].encode('utf-8')).hexdigest()
        # doc_id = f"{config.CHROMA_COLLECTION_NAME}_{doc['metadata']['filename']}_{content_hash[:8]}"
        doc_ids.append(doc_id)

    if not doc_contents:
        print("No valid documents found to process after filtering.")
        return False

    try:
        # Generate embeddings for all document contents
        embeddings = embedding_model.encode(doc_contents, convert_to_tensor=False)
        
        # Ensure embeddings are a list of lists of floats
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

        # Add items to the collection
        # Note: If IDs conflict, ChromaDB's default behavior is to update/upsert.
        collection.add(
            ids=doc_ids,
            embeddings=embeddings_list,
            documents=doc_contents,
            metadatas=doc_metadatas
        )
        print(f"Successfully added/updated {len(doc_ids)} documents in collection '{config.CHROMA_COLLECTION_NAME}'.")
        print(f"Total items in collection: {collection.count()}")
        return True
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        return False


def query_documents(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Queries the ChromaDB collection for documents similar to the query_text.

    Args:
        query_text (str): The text to query the collection with.
        n_results (int): The number of similar documents to retrieve.

    Returns:
        list[dict]: A list of retrieved documents, where each document is a dictionary
                    containing 'content' and 'metadata'. Returns an empty list if
                    an error occurs or no results are found.
    """
    if collection is None or embedding_model is None:
        print("Error: ChromaDB collection or SentenceTransformer model is not initialized. Cannot query documents.")
        return []
    
    if not query_text:
        print("Error: Query text is empty.")
        return []
        
    try:
        # Generate an embedding for the query text
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
        
        # Ensure query_embedding is a list of floats (it will be if input query_text is a single string)
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        
        results = collection.query(
            query_embeddings=[query_embedding_list], # Must be a list of embeddings
            n_results=min(n_results, collection.count()), # Cannot request more results than items in collection
            include=['documents', 'metadatas', 'distances']
        )
        
        retrieved_docs = []
        if results and results.get('documents') and results.get('metadatas'):
            # ChromaDB returns lists of lists for documents, metadatas, etc.
            # Since we query with a single embedding, we expect results structure like:
            # {'documents': [['doc1_content', 'doc2_content']], 'metadatas': [[{'f1':'v1'}, {'f2':'v2'}]] ...}
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })
            print(f"Query successful. Retrieved {len(retrieved_docs)} documents for: '{query_text}'")
        else:
            print(f"No results found or unexpected result structure for query: '{query_text}'")
            
        return retrieved_docs
    except Exception as e:
        print(f"Error querying ChromaDB with text '{query_text}': {e}")
        return []

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    print("\n--- Testing ChromaDB Vector Store with React Components ---")

    if client is None or collection is None or embedding_model is None:
        print("Critical component (ChromaDB client, collection, or embedding model) failed to initialize.")
        print("Cannot run tests for vector_store.py.")
    else:
        # 1. Load React components using document_processor
        # Need to import it first.
        # Ensure document_processor.py is in the Python path.
        # If running `python backend/vector_store.py` from `/app`, then `from .document_processor` is correct.
        try:
            from .document_processor import load_and_chunk_react_components
        except ImportError:
            print("Error: Could not import 'load_and_chunk_react_components' from '.document_processor'.")
            # This fallback might be problematic if not run as part of the package.
            # For robust testing, it's better to run `python -m backend.vector_store` from /app
            print("Attempting direct import (may fail if not in appropriate CWD or PYTHONPATH).")
            load_and_chunk_react_components = None # Or try `from document_processor ...` if that was intended for some contexts

            if load_and_chunk_react_components is None: # Try the old fallback as a last resort for existing test structure
                try:
                    from document_processor import load_and_chunk_react_components
                    print("Imported 'load_and_chunk_react_components' from 'document_processor' (local path - ensure CWD is backend/).")
                except ImportError:
                    print("Failed to import 'load_and_chunk_react_components' from local path 'document_processor' too.")
                    load_and_chunk_react_components = None


        if load_and_chunk_react_components:
            # The path to components should be relative to CWD or absolute.
            # Default in `load_and_chunk_react_components` is "backend/retrieved_components/"
            # If running `python backend/vector_store.py` from `/app`, this path is correct.
            # If running from `backend/`, it should be "retrieved_components/"
            # For consistency, let's define it here assuming run from /app
            components_path = "backend/retrieved_components/"
            if not os.path.isdir(components_path):
                 # Try path relative to this script file (if running from backend/)
                 script_dir = os.path.dirname(__file__)
                 alt_components_path = os.path.join(script_dir, "retrieved_components")
                 if os.path.isdir(alt_components_path):
                     components_path = alt_components_path
                 else:
                     print(f"Warning: Components directory '{components_path}' (and alternate '{alt_components_path}') not found.")
                     print("Please ensure the components directory exists and path is correct relative to CWD.")


            print(f"\n--- Loading React components from: {os.path.abspath(components_path)} ---")
            react_documents = load_and_chunk_react_components(components_dir=components_path)

            if react_documents:
                print(f"Loaded {len(react_documents)} React component documents.")
                
                # 2. Add them to ChromaDB
                print("\n--- Adding React components to ChromaDB ---")
                # Clear the collection for a clean test run each time (optional)
                print(f"Current collection count before adding: {collection.count()}")
                # To clear:
                # client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
                # collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
                # print(f"Collection '{config.CHROMA_COLLECTION_NAME}' cleared and recreated. Count: {collection.count()}")

                add_success = add_documents(react_documents)
                if add_success:
                    print(f"Successfully added documents. Collection count: {collection.count()}")
                else:
                    print("Failed to add documents to ChromaDB.")

                # 3. Perform a sample query
                if collection.count() > 0:
                    print("\n--- Querying ChromaDB for React components ---")
                    sample_query = "a button component"
                    query_results = query_documents(sample_query, n_results=2)
                    
                    if query_results:
                        print(f"\nFound {len(query_results)} results for query: '{sample_query}':")
                        for i, doc in enumerate(query_results):
                            print(f"\nResult {i+1}:")
                            print(f"  Filename: {doc['metadata'].get('filename', 'N/A')}")
                            print(f"  Distance: {doc.get('distance', 'N/A'):.4f}")
                            print(f"  Content (first 80 chars): {doc['content'][:80]}...")
                    else:
                        print(f"No results found for query: '{sample_query}'")
                else:
                    print("\nSkipping query test as collection is empty or adding failed.")
            else:
                print("\nNo React component documents loaded, skipping add and query tests.")
        else:
            print("\n`load_and_chunk_react_components` function not available. Skipping tests.")
