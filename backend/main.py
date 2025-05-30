from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
from pydantic import BaseModel, Field # For request and response data modeling
import os # For operating system dependent functionality, like path joining
from contextlib import asynccontextmanager # For lifespan events

# Import functionalities from other backend modules
from . import config # Import config
from .vector_store import (
    # process_docs_for_rag is removed, replaced by specific component processing
    query_documents, # For querying with user message
    add_documents,   # For adding new (React component) documents
    collection as chromadb_collection, # For checking initialization status
    client as chromadb_client, # For checking initialization status
    embedding_model as vs_embedding_model # To check if vector_store's model loaded
)
# generate_query_embedding from document_processor is removed, vector_store.query_documents handles it
from .document_processor import load_and_chunk_react_components
# Import the LLM generation function and the pipeline object (to check its status)
# generate_response_from_context and generator_pipeline were for a previous HuggingFace-based RAG chat.
# These have been removed from llm_generator.py, which now focuses on OpenAI for code generation.
# Commenting out these imports and related code in /chat and lifespan for now.
# from .llm_generator import generate_response_from_context, generator_pipeline as llm_pipeline_global
# For the new Figma to React generation service:
from .react_generator_service import generate_react_for_figma_node
# For Pydantic model field type
from typing import List


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
        print(f"Collection '{config.CHROMA_COLLECTION_NAME}' (expected: {chromadb_collection.name}) has {chromadb_collection.count()} items at startup.")

    # Startup: Check Vector Store's embedding model
    if vs_embedding_model is None:
        print("WARNING: Vector Store's SentenceTransformer model was not initialized at startup.")
    else:
        print("FastAPI startup: Vector Store's SentenceTransformer model appears to be initialized.")

    # Startup: Check LLM pipeline status (Commented out as pipeline was removed)
    # if llm_pipeline_global is None:
    #     print("WARNING: LLM pipeline (HuggingFace) was not initialized at startup. General RAG chat might not work.")
    # else:
    #     print("FastAPI startup: LLM generation pipeline (HuggingFace) appears to be initialized.")
    print("FastAPI startup: Note - HuggingFace RAG chat functionality is currently disabled due to llm_generator.py changes.")
    
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
    # This model might be deprecated if /process-documents is removed or changed.

class FigmaToReactRequest(BaseModel):
    figma_file_key: str = Field(..., description="The Figma file key.")
    node_ids: List[str] = Field(..., description="A list of Figma node IDs to generate React components for.")


# --- API Endpoints ---
@app.get("/")
async def read_root():
    """ Root endpoint providing a welcome message. """
    return {"message": "Welcome to the RAG chatbot backend!"}

# The old /process-documents endpoint might be deprecated or removed in favor of /process-components
# For now, it's left here but might need adjustment if `process_docs_for_rag` was removed from vector_store.
# Based on previous steps, `process_and_embed_documents` (aliased as `process_docs_for_rag`) was indeed removed from vector_store.py.
# So, this endpoint will fail if called. It should be removed or updated.
# For now, let's comment it out to avoid errors during startup/testing of other endpoints.
"""
@app.post("/process-documents")
async def process_documents_endpoint(request_body: ProcessRequest):
    # This endpoint is now problematic because `process_docs_for_rag` was removed.
    # It needs to be updated or removed.
    # ... (original code commented out) ...
    pass
"""

@app.post("/process-components")
async def process_components_endpoint():
    """
    Endpoint to trigger the processing of React component files.
    Components are loaded from a predefined directory, chunked (each file is a chunk),
    embedded, and stored in ChromaDB.
    """
    if chromadb_collection is None or vs_embedding_model is None:
        raise HTTPException(status_code=500, detail="ChromaDB or SentenceTransformer model not initialized.")

    try:
        # Default components directory is used from document_processor.py
        print("Processing React components...")
        component_documents = load_and_chunk_react_components()

        if not component_documents:
            print("No components found or loaded.")
            # It's not an error if no components are found; could be an empty dir.
            # Consider if this should be an HTTPException or a success with 0 processed.
            return {"message": "No React components found or loaded.", "processed_count": 0}

        print(f"Found {len(component_documents)} component documents to process.")

        success = add_documents(component_documents)

        if success:
            return {
                "message": "React components processed and added to vector store successfully.",
                "processed_count": len(component_documents),
                "collection_total_items": chromadb_collection.count()
            }
        else:
            # add_documents logs errors internally.
            raise HTTPException(status_code=500, detail="Failed to add component documents to vector store.")

    except Exception as e:
        print(f"Error during /process-components: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    """
    Endpoint to handle chat requests.
    It takes a user's message, generates an embedding (using vector_store's model),
    retrieves relevant context from ChromaDB (via vector_store.query_documents),
    and then uses an LLM to generate a response based on the context.
    """
    user_message = chat_message.message
    
    if chromadb_collection is None or vs_embedding_model is None:
        raise HTTPException(status_code=500, detail="ChromaDB or SentenceTransformer model not initialized. Cannot process chat.")

    # if llm_pipeline_global is None: # Old HuggingFace pipeline check
    #     print("LLM pipeline is not available. Check model loading in llm_generator.py.")
    #     raise HTTPException(status_code=500, detail="LLM generation pipeline is not available. Cannot generate response.")
    # For now, the /chat endpoint is effectively disabled if it relied on the removed HF pipeline.
    # A new chat implementation would be needed, possibly using OpenAI.
    # Returning a placeholder message.
    print("Warning: The /chat endpoint is currently using a placeholder response as its LLM integration needs update.")
    # Simulate fetching context, but don't call LLM
    retrieved_docs_with_metadata = query_documents(
        query_text=user_message,
        n_results=3
    )
    retrieved_doc_chunks = []
    if retrieved_docs_with_metadata:
        for doc in retrieved_docs_with_metadata:
            retrieved_doc_chunks.append(doc['content'])
    
    return {
        "user_query": user_message,
        "retrieved_context_chunks": retrieved_doc_chunks,
        "llm_response": "Chat functionality is under maintenance. Please try again later."
    }

    # Step 1 & 2: Query ChromaDB using the query_documents function from vector_store
    # This function handles embedding the query_text and querying the collection.
    # It returns a list of dicts, each with 'content', 'metadata', 'distance'.
    retrieved_docs_with_metadata = query_documents(
        query_text=user_message,
        n_results=3 # Retrieve the top 3 most relevant document chunks
    )

    retrieved_doc_chunks = []
    if retrieved_docs_with_metadata:
        for doc in retrieved_docs_with_metadata:
            retrieved_doc_chunks.append(doc['content']) # Extract just the content for the LLM
    
    print(f"Retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message}'.")
    if retrieved_doc_chunks:
        for i, chunk in enumerate(retrieved_doc_chunks):
            print(f"Chunk {i+1}: {chunk[:200]}...")

    # Step 3: Generate a response using the LLM with the retrieved context (Old code, commented out)
    # llm_response_text = generate_response_from_context(
    #     query=user_message,
    #     context_chunks=retrieved_doc_chunks, # Pass the list of retrieved document strings
    #     max_context_length=1500, # Max characters for context part of the prompt
    #     max_new_tokens=100       # Max new tokens for the LLM to generate
    # )

    # Return the user's query, the retrieved context (for transparency/debugging), and the LLM's response
    # return {
    #     "user_query": user_message,
    #     "retrieved_context_chunks": retrieved_doc_chunks,
    #     "llm_response": llm_response_text # This is the key field for the frontend
    # }


@app.post("/generate-react-from-figma")
async def generate_react_from_figma_endpoint(request: FigmaToReactRequest):
    """
    Endpoint to generate React components from specified Figma nodes.
    Accepts a Figma file key and a list of node IDs.
    """
    print(f"Received request to generate React components for file_key: {request.figma_file_key}, node_ids: {request.node_ids}")

    results = []
    for node_id in request.node_ids:
        print(f"Processing node_id: {node_id}...")
        try:
            # Each call to generate_react_for_figma_node performs the full RAG pipeline for one node
            node_result = generate_react_for_figma_node(
                file_key=request.figma_file_key,
                node_id=node_id
            )
            results.append(node_result)
            print(f"Successfully processed node_id: {node_id}. Result keys: {list(node_result.keys())}")
        except Exception as e:
            # This catches unexpected errors within the loop for a specific node
            # generate_react_for_figma_node itself should return error dicts for known issues
            print(f"An unexpected error occurred while processing node_id {node_id}: {e}")
            results.append({
                "figma_node_id": node_id,
                "error": "An unexpected error occurred during processing.",
                "details": str(e)
            })

    print(f"Finished processing all requested node_ids. Total results: {len(results)}")
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    # Standard Uvicorn entry point when running the module directly.
    # Host 0.0.0.0 makes it accessible externally if needed (e.g. from host to Docker container).
    # Port 8000 is a common default.
    # reload=True can be useful for development but is set to False for this context.
    print("Attempting to start Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
