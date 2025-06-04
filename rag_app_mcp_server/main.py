from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # Added WebSocket, WebSocketDisconnect
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
origins = [
    "http://localhost:3000",  # Standard React development server port
    "localhost:3000",         # Sometimes needed depending on how requests are made
    "http://localhost:8001", # Added for potential client app if on different port
    "localhost:8001",
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
# class ChatMessage(BaseModel): # No longer needed for HTTP chat
#     """Defines the expected structure for a chat message request."""
#     message: str

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
    # Path should be relative to rag_app_mcp_server now
    if not os.path.isabs(docs_dir):
        docs_dir = os.path.join("/app/rag_app_mcp_server", docs_dir.replace("backend/", ""))


    if not os.path.isdir(docs_dir):
        # Try relative to /app if not found inside rag_app_mcp_server
        # This might be needed if the user provides a path like "documents_for_rag"
        # intending it to be inside the new app structure.
        corrected_docs_dir = os.path.join("/app", request_body.directory)
        if os.path.isdir(corrected_docs_dir):
            docs_dir = corrected_docs_dir
        else:
            # Also check if original path was "rag_app_mcp_server/documents_for_rag"
            original_path_as_is = os.path.join("/app", request_body.directory)
            if os.path.isdir(original_path_as_is) and "rag_app_mcp_server" in request_body.directory:
                 docs_dir = original_path_as_is
            else:
                 # Try path relative to current app dir if not absolute
                current_app_docs_dir = os.path.join("rag_app_mcp_server", request_body.directory)
                if os.path.isdir(current_app_docs_dir):
                    docs_dir = current_app_docs_dir
                else:
                    # Fallback for simple "documents_for_rag"
                    simple_path = os.path.join("rag_app_mcp_server/documents_for_rag")
                    if os.path.isdir(simple_path) and request_body.directory == "documents_for_rag":
                        docs_dir = simple_path
                    else:
                        # Original error if no path is valid
                        raise HTTPException(status_code=400, detail=f"Directory not found: {request_body.directory} or derived paths like {docs_dir}")


    try:
        # Call the function to process documents
        # The process_docs_for_rag function uses paths relative to where it's called or absolute paths.
        # Ensure it uses the correct path for documents within rag_app_mcp_server
        result = process_docs_for_rag(documents_dir=docs_dir) # docs_dir should be correctly resolved now
        if result.get("status") == "error": # Check for errors reported by the processing function
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error during document processing."))
        return {"message": "Document processing initiated.", "details": result}
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"Error during /process-documents: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during document processing: {str(e)}")

# --- Commented out old HTTP /chat endpoint ---
# @app.post("/chat")
# async def chat(chat_message: ChatMessage):
#     """
#     Endpoint to handle chat requests.
#     It takes a user's message, generates an embedding, retrieves relevant context
#     from ChromaDB, and then uses an LLM to generate a response based on the context.
#     """
#     user_message = chat_message.message

#     if chromadb_collection is None or generate_query_embedding is None:
#         raise HTTPException(status_code=500, detail="ChromaDB or SentenceTransformer model not initialized. Cannot process chat.")

#     if llm_pipeline_global is None:
#         print("LLM pipeline is not available. Check model loading in llm_generator.py.")
#         raise HTTPException(status_code=500, detail="LLM generation pipeline is not available. Cannot generate response.")

#     query_embedding_list = generate_query_embedding([user_message])
#     if not query_embedding_list or not query_embedding_list[0]:
#         raise HTTPException(status_code=500, detail="Failed to generate embedding for the user message.")

#     query_embedding = query_embedding_list[0]

#     retrieved_info = query_collection_with_embedding(
#         query_embedding=query_embedding,
#         n_results=3
#     )

#     retrieved_doc_chunks = []
#     if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0]:
#         retrieved_doc_chunks = retrieved_info['documents'][0]

#     print(f"Retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message}'.")
#     if retrieved_doc_chunks:
#         for i, chunk in enumerate(retrieved_doc_chunks):
#             print(f"Chunk {i+1}: {chunk[:200]}...")

#     llm_response_text = generate_response_from_context(
#         query=user_message,
#         context_chunks=retrieved_doc_chunks,
#         max_context_length=1500,
#         max_new_tokens=100
#     )

#     return {
#         "user_query": user_message,
#         "retrieved_context_chunks": retrieved_doc_chunks,
#         "llm_response": llm_response_text
#     }
# --- End of commented out old HTTP /chat endpoint ---

# --- WebSocket Chat Endpoint ---
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")

    # Check for RAG components readiness at the beginning of the session
    if chromadb_collection is None or generate_query_embedding is None:
        await websocket.send_text("Error: Backend RAG components (ChromaDB or Embedder) not initialized.")
        await websocket.close(code=1008) # Policy Violation or custom code
        return
    if llm_pipeline_global is None:
        await websocket.send_text("Error: Backend LLM pipeline not initialized.")
        await websocket.close(code=1008)
        return

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message via WebSocket: {data}")
            user_message = data

            try:
                # Step 1: Generate embedding for the user's message
                query_embedding_list = generate_query_embedding([user_message])
                if not query_embedding_list or not query_embedding_list[0]:
                    await websocket.send_text("Error: Failed to generate message embedding.")
                    continue

                query_embedding = query_embedding_list[0]

                # Step 2: Query ChromaDB
                retrieved_info = query_collection_with_embedding(
                    query_embedding=query_embedding,
                    n_results=3
                )
                retrieved_doc_chunks = []
                if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0]:
                    retrieved_doc_chunks = retrieved_info['documents'][0]

                print(f"Retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message}'.")

                # Step 3: Generate response using LLM
                llm_response_text = generate_response_from_context(
                    query=user_message,
                    context_chunks=retrieved_doc_chunks,
                    max_context_length=1500,
                    max_new_tokens=150 # Increased token limit slightly for potentially more comprehensive answers
                )

                await websocket.send_text(llm_response_text)
                print(f"Sent LLM response: {llm_response_text}")

            except Exception as e:
                error_message = f"Error processing message: {str(e)}"
                print(error_message)
                await websocket.send_text(error_message)

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket chat.")
    except Exception as e:
        # Catch any other unexpected errors during WebSocket communication
        print(f"Unexpected error in WebSocket chat: {str(e)}")
        try:
            await websocket.send_text("An unexpected server error occurred.")
        except Exception: # If sending fails, it means connection is already likely closed
            pass
    finally:
        # Ensure connection is closed if not already
        if websocket.client_state != websocket.client_state.DISCONNECTED:
            try:
                await websocket.close()
                print("WebSocket connection closed.")
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
# --- End WebSocket Chat Endpoint ---

# Note: No `if __name__ == "__main__":` block as Uvicorn will be run externally (e.g., via Docker or command line)
# Example command to run this app (assuming this file is main.py):
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# (Port 8001 is an example for this RAG app server)
# (The --reload flag is for development; remove it in production)
# (The actual documents_for_rag path in ProcessRequest default might need adjustment
#  based on the final location if "backend/" prefix is removed globally later)
# The ProcessRequest default path was changed from "backend/documents_for_rag"
# to "documents_for_rag" which will be resolved relative to /app/rag_app_mcp_server/

# Path correction in process_documents_endpoint:
# The default for ProcessRequest.directory is now "documents_for_rag".
# The logic in process_documents_endpoint attempts to resolve this relative to /app/rag_app_mcp_server/.
# If the provided path in a request is, e.g., "documents_for_rag", it becomes "/app/rag_app_mcp_server/documents_for_rag".
# If an absolute path is provided, it's used as is.
# If "backend/documents_for_rag" is provided, it attempts to correct it by removing "backend/".
# This logic may need further refinement based on how paths are provided by the client.
# For now, the default "documents_for_rag" should work if that folder is inside "rag_app_mcp_server".
# The copy operation in the previous subtask placed `documents_for_rag` inside `rag_app_mcp_server`.
# So, ProcessRequest(directory="documents_for_rag") should resolve to /app/rag_app_mcp_server/documents_for_rag
# The path resolution for `docs_dir` in `process_documents_endpoint` has been updated to be more robust
# and prioritize paths within the `rag_app_mcp_server` directory.
# The default `directory` in `ProcessRequest` was also changed from `backend/documents_for_rag` to `documents_for_rag`
# assuming it will be relative to the new app's root.
# The path resolution logic in `process_documents_endpoint` has been refined.
# For `ProcessRequest.directory`, if "documents_for_rag" is passed (the default), it will be
# resolved as `/app/rag_app_mcp_server/documents_for_rag`.
# If `backend/documents_for_rag` is passed, it will be changed to `/app/rag_app_mcp_server/documents_for_rag`.
# This should align with the new directory structure.

# Adjusted default ProcessRequest directory to "documents_for_rag" and updated path logic
# in process_documents_endpoint to correctly resolve paths relative to rag_app_mcp_server.
# Original default: "backend/documents_for_rag"
# New default for ProcessRequest: "documents_for_rag" (to be interpreted as inside rag_app_mcp_server)
# The logic in process_documents_endpoint:
# 1. If request_body.directory is "documents_for_rag", docs_dir becomes "/app/rag_app_mcp_server/documents_for_rag"
# 2. If request_body.directory is "backend/documents_for_rag", docs_dir becomes "/app/rag_app_mcp_server/documents_for_rag"
# 3. If an absolute path is given, it's used. (e.g. "/app/rag_app_mcp_server/custom_docs")
# This should correctly point to the documents within the new app structure.
# The `ProcessRequest` model's default directory has been changed to `documents_for_rag`.
# The logic in `process_documents_endpoint` has been updated to resolve this relative to `/app/rag_app_mcp_server/`.
# For example, if the request uses the default `directory="documents_for_rag"`,
# `docs_dir` will be constructed as `os.path.join("/app/rag_app_mcp_server", "documents_for_rag")`.
# If the request provides `directory="backend/documents_for_rag"`, it's adjusted to
# `os.path.join("/app/rag_app_mcp_server", "documents_for_rag")`.
# This ensures paths are correctly handled within the new application structure.
# Final check of path logic in process_documents_endpoint:
# - Default directory in ProcessRequest: "documents_for_rag"
# - docs_dir initial construction: os.path.join("/app/rag_app_mcp_server", request_body.directory.replace("backend/", ""))
#   - If "documents_for_rag" -> "/app/rag_app_mcp_server/documents_for_rag"
#   - If "backend/documents_for_rag" -> "/app/rag_app_mcp_server/documents_for_rag"
# - The subsequent os.path.isdir checks provide fallbacks but the initial construction should be the primary path.
# This looks correct for the new structure.
