from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
from pydantic import BaseModel # For request and response data modeling
import os # For operating system dependent functionality, like path joining
from contextlib import asynccontextmanager # For lifespan events

# Import functionalities from other backend modules
from vector_store import (
    process_and_embed_documents as process_docs_for_rag, # Renamed for clarity
    query_collection_with_embedding, # For querying with user message embedding
    collection as chromadb_collection, # For checking initialization status
    client as chromadb_client # For checking initialization status
)
from document_processor import generate_embeddings as generate_query_embedding # For user messages
# Import the LLM generation function and the pipeline object (to check its status)
from llm_generator import generate_response_from_context, generator_pipeline as llm_pipeline_global

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    """
    # Startup: Check ChromaDB client and collection status
    if chromadb_client is None or chromadb_collection is None:
        print("WARNING: ChromaDB client or collection was not initialized at startup.")
    else:
        print("FastAPI startup: ChromaDB client and collection appear to be initialized.")
        print(f"Collection '{chromadb_collection.name}' has {chromadb_collection.count()} items at startup.")
    
    # Startup: Check LLM pipeline status
    if llm_pipeline_global is None:
        print("WARNING: LLM pipeline was not initialized at startup. LLM generation will not work.")
    else:
        print("FastAPI startup: LLM generation pipeline appears to be initialized.")
    
    yield # Application runs after this point
    
    # Shutdown: (Optional) Add any cleanup code here if needed in the future
    print("FastAPI shutdown: Application is shutting down.")

# Initialize the FastAPI application instance
app = FastAPI(
    title="RAG Chatbot Backend",
    description="API for the Retrieval Augmented Generation (RAG) chatbot.",
    version="0.1.0",
    lifespan=lifespan # Use the new lifespan context manager
)

# --- CORS Configuration ---
# Define allowed origins for Cross-Origin Resource Sharing.
# This is crucial for allowing the frontend (running on localhost:3000)
# to communicate with the backend (running on localhost:8000).
origins = [
    "http://localhost:3000",  # Standard React development server port
    "localhost:3000",         # Sometimes needed depending on how requests are made
    # Add other origins if your frontend is served from different URLs in production/staging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of origins that are allowed to make requests
    allow_credentials=True, # Allow cookies to be included in requests
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- End CORS Configuration ---

# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    """Defines the expected structure for a chat message request."""
    message: str

class ProcessRequest(BaseModel):
    """Defines the expected structure for a document processing request."""
    directory: str = "backend/documents_for_rag" # Default directory for processing documents

# --- API Endpoints ---
@app.get("/")
async def read_root():
    """ Root endpoint providing a welcome message. """
    return {"message": "Welcome to the RAG chatbot backend!"}

@app.post("/process-documents")
async def process_documents_endpoint(request_body: ProcessRequest):
    """
    Endpoint to trigger the processing of documents from a specified directory.
    Documents are loaded, chunked, embedded, and stored in ChromaDB.
    """
    # Check if critical components are available (ChromaDB and embedding model via generate_query_embedding)
    if chromadb_collection is None or generate_query_embedding is None: # generate_query_embedding check also implies sentence transformer model loaded
        raise HTTPException(status_code=500, detail="ChromaDB or SentenceTransformer model not initialized.")

    docs_dir = request_body.directory
    # Ensure the path is absolute or resolve it relative to /app (common in Docker)
    if not os.path.isabs(docs_dir):
        docs_dir = os.path.join("/app", docs_dir) 

    if not os.path.isdir(docs_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {docs_dir}")

    try:
        # Call the function to process documents
        result = process_docs_for_rag(documents_dir=docs_dir)
        if result.get("status") == "error": # Check for errors reported by the processing function
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error during document processing."))
        return {"message": "Document processing initiated.", "details": result}
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"Error during /process-documents: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during document processing: {str(e)}")


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    """
    Endpoint to handle chat requests.
    It takes a user's message, generates an embedding, retrieves relevant context
    from ChromaDB, and then uses an LLM to generate a response based on the context.
    """
    user_message = chat_message.message
    
    # Check if critical components are available
    if chromadb_collection is None or generate_query_embedding is None:
        raise HTTPException(status_code=500, detail="ChromaDB or SentenceTransformer model not initialized. Cannot process chat.")

    if llm_pipeline_global is None:
        # This check is important because LLM model loading might fail (e.g. due to resource constraints)
        # The llm_generator module attempts to load it on import; if it fails, llm_pipeline_global will be None.
        print("LLM pipeline is not available. Check model loading in llm_generator.py.")
        raise HTTPException(status_code=500, detail="LLM generation pipeline is not available. Cannot generate response.")

    # Step 1: Generate embedding for the user's message
    # generate_query_embedding expects a list of texts and returns a list of embeddings.
    query_embedding_list = generate_query_embedding([user_message])
    if not query_embedding_list or not query_embedding_list[0]: # Check if embedding generation failed
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the user message.")
    
    query_embedding = query_embedding_list[0] # Get the embedding for the single user message

    # Step 2: Query ChromaDB using the embedding to get relevant context
    retrieved_info = query_collection_with_embedding(
        query_embedding=query_embedding,
        n_results=3 # Retrieve the top 3 most relevant document chunks
    )

    retrieved_doc_chunks = []
    # ChromaDB query results for 'documents' is a list of lists.
    if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0]:
        retrieved_doc_chunks = retrieved_info['documents'][0] # Get the list of chunks for the first query
    
    print(f"Retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message}'.")
    if retrieved_doc_chunks: # Log retrieved chunks for debugging
        for i, chunk in enumerate(retrieved_doc_chunks):
            print(f"Chunk {i+1}: {chunk[:200]}...") # Print first 200 characters of each chunk

    # Step 3: Generate a response using the LLM with the retrieved context
    llm_response_text = generate_response_from_context(
        query=user_message,
        context_chunks=retrieved_doc_chunks, # Pass the list of retrieved document strings
        max_context_length=1500, # Max characters for context part of the prompt
        max_new_tokens=100       # Max new tokens for the LLM to generate
    )

    # Return the user's query, the retrieved context (for transparency/debugging), and the LLM's response
    return {
        "user_query": user_message,
        "retrieved_context_chunks": retrieved_doc_chunks, 
        "llm_response": llm_response_text # This is the key field for the frontend
    }
