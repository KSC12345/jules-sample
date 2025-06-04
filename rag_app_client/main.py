from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os # Ensure os is imported
from contextlib import asynccontextmanager
import websockets

# Import functionalities from other backend modules (local RAG components)
from vector_store import (
    process_and_embed_documents as process_docs_for_rag,
    query_collection_with_embedding,
    collection as chromadb_collection,
    client as chromadb_client
)
from document_processor import generate_embeddings as generate_query_embedding
from llm_generator import generate_response_from_context, generator_pipeline as llm_pipeline_global

# --- Configuration for MCP Server ---
# MCP_SERVER_URL will be taken from environment variable if set, otherwise defaults to localhost:8001
# This allows Docker Compose to set it for inter-container communication,
# while local execution can still use the default.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "ws://localhost:8001/ws/chat")

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check local ChromaDB client and collection status
    if chromadb_client is None or chromadb_collection is None:
        print("RAG App 2 WARNING: Local ChromaDB client or collection not initialized.")
    else:
        print(f"RAG App 2 ({app.title}): Local ChromaDB initialized. Collection '{chromadb_collection.name}' has {chromadb_collection.count()} items.")

    # Startup: Check local LLM pipeline status
    if llm_pipeline_global is None:
        print(f"RAG App 2 ({app.title}) WARNING: Local LLM pipeline not initialized.")
    else:
        print(f"RAG App 2 ({app.title}): Local LLM pipeline initialized.")

    # Print the MCP_SERVER_URL that will be used
    print(f"RAG App 2 ({app.title}): Connecting to MCP Server at {MCP_SERVER_URL}")

    yield
    print(f"RAG App 2 ({app.title}): Application shutting down.")

# Initialize the FastAPI application instance for RAG App 2
app = FastAPI(
    title="RAG App 2 (Client with Forwarding Logic)",
    description="Handles queries locally or forwards them to an MCP RAG server.",
    version="0.1.0",
    lifespan=lifespan
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000", # Example frontend
    "localhost:3000",
    "http://localhost:8001", # MCP Server (RAG App 1) - for direct access if needed
    "localhost:8001",
    "http://localhost:8002", # This app (RAG App 2)
    "localhost:8002",
    # Add service names if access is direct via Docker overlay network and specific hostnames
    "http://rag_mcp_server_container:8000", # For potential direct calls within Docker network
    "http://rag_client_app_container:8000", # For potential direct calls within Docker network
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ProcessRequest(BaseModel):
    directory: str = "documents_for_rag" # Default directory within rag_app_client

# --- Decision Function ---
def should_forward_to_mcp(query: str) -> bool:
    """
    Determines if a query should be forwarded to the MCP server.
    For now, uses a simple keyword match.
    """
    return "FORWARD_TO_MCP" in query.upper()

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to RAG App 2 (Client with Forwarding Logic)!"}

@app.post("/process-documents")
async def process_documents_endpoint(request_body: ProcessRequest):
    if chromadb_collection is None or generate_query_embedding is None:
        raise HTTPException(status_code=500, detail="Local ChromaDB or SentenceTransformer model not initialized for RAG App 2.")

    docs_dir_input = request_body.directory

    if not os.path.isabs(docs_dir_input):
        if docs_dir_input.startswith("rag_app_client/"):
            # Path like "rag_app_client/documents_for_rag" -> "/app/documents_for_rag" (if WORKDIR is /app)
            # This case might be less common if WORKDIR is /app/rag_app_client
            # Assuming WORKDIR /app as per Dockerfile standard
            base_dir = "/app"
            resolved_path = os.path.join(base_dir, docs_dir_input.split('/', 1)[1] if '/' in docs_dir_input else "")
        else:
            # Path like "documents_for_rag" -> "/app/documents_for_rag"
            # This assumes the app's specific files are directly under /app, which means
            # the Dockerfile for rag_app_client should be COPY . /app NOT COPY . /app/rag_app_client
            # Given the docker-compose volume mounts `./rag_app_client:/app`, this path should be /app/documents_for_rag
            # The CHROMA_DB_PATH is /app/chroma_data, so documents should also be /app/documents_for_rag
            resolved_path = os.path.join("/app", docs_dir_input)
    else:
        resolved_path = docs_dir_input

    # Correcting path resolution based on Dockerfile WORKDIR /app and docker-compose volume mount ./rag_app_client:/app
    # The files from ./rag_app_client (like main.py, documents_for_rag) are directly in /app within the container.
    # So, "documents_for_rag" should resolve to "/app/documents_for_rag".

    current_script_dir = os.path.dirname(os.path.abspath(__file__)) # This is /app in the container

    if not os.path.isabs(docs_dir_input):
        # docs_dir_input is "documents_for_rag"
        docs_dir = os.path.join(current_script_dir, docs_dir_input) # /app/documents_for_rag
    else:
        docs_dir = docs_dir_input # Absolute path provided

    if not os.path.isdir(docs_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found for RAG App 2: {docs_dir} (resolved from input: {docs_dir_input})")

    try:
        result = process_docs_for_rag(documents_dir=docs_dir) # documents_dir is now /app/documents_for_rag
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error during document processing in RAG App 2."))
        return {"message": "Document processing for RAG App 2 initiated.", "details": result}
    except Exception as e:
        print(f"Error during /process-documents for RAG App 2: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred for RAG App 2: {str(e)}")

# --- WebSocket Chat Endpoint for RAG App 2 ---
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("RAG App 2: WebSocket connection established.")

    local_rag_ready = chromadb_collection is not None and \
                      generate_query_embedding is not None and \
                      llm_pipeline_global is not None

    if not local_rag_ready:
        print("RAG App 2: Warning - Local RAG components are not fully initialized. Local processing may fail.")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"RAG App 2: Received message: {data}")
            user_message = data

            if should_forward_to_mcp(user_message):
                print(f"RAG App 2: Query '{user_message}' contains forwarding keyword. Attempting to forward to MCP server at {MCP_SERVER_URL}.")
                try:
                    async with websockets.connect(MCP_SERVER_URL) as mcp_ws:
                        await mcp_ws.send(user_message)
                        mcp_response = await mcp_ws.recv()
                        print(f"RAG App 2: Received response from MCP: {mcp_response}")
                        await websocket.send_text(f"Forwarded to MCP, Response: {mcp_response}")
                except websockets.exceptions.ConnectionClosedError as e:
                    error_msg = f"RAG App 2: MCP server connection closed. {e}"
                    print(error_msg)
                    await websocket.send_text(f"Error: {error_msg}")
                except ConnectionRefusedError as e:
                    error_msg = f"RAG App 2: Could not connect to MCP server at {MCP_SERVER_URL}. Connection refused. {e}"
                    print(error_msg)
                    await websocket.send_text(f"Error: {error_msg}")
                except Exception as e:
                    error_msg = f"RAG App 2: Error during forwarding to MCP: {str(e)}"
                    print(error_msg)
                    await websocket.send_text(f"Error: {error_msg}")
            else:
                print(f"RAG App 2: Query '{user_message}' does not contain forwarding keyword. Processing locally.")
                if not local_rag_ready:
                    await websocket.send_text("Error: Local RAG components not ready for processing.")
                    continue

                try:
                    query_embedding_list = generate_query_embedding([user_message])
                    if not query_embedding_list or not query_embedding_list[0]:
                        await websocket.send_text("Error: Failed to generate message embedding locally.")
                        continue

                    query_embedding = query_embedding_list[0]

                    retrieved_info = query_collection_with_embedding(query_embedding=query_embedding, n_results=3)
                    retrieved_doc_chunks = []
                    if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0]:
                        retrieved_doc_chunks = retrieved_info['documents'][0]

                    print(f"RAG App 2: Locally retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message}'.")

                    llm_response_text = generate_response_from_context(
                        query=user_message,
                        context_chunks=retrieved_doc_chunks,
                        max_context_length=1500,
                        max_new_tokens=150
                    )

                    await websocket.send_text(f"Locally Processed: {llm_response_text}")
                    print(f"RAG App 2: Sent local LLM response: {llm_response_text}")

                except Exception as e:
                    error_message = f"RAG App 2: Error processing message locally: {str(e)}"
                    print(error_message)
                    await websocket.send_text(error_message)

    except WebSocketDisconnect:
        print("RAG App 2: Client disconnected from WebSocket chat.")
    except Exception as e:
        print(f"RAG App 2: Unexpected error in WebSocket chat: {str(e)}")
        try:
            await websocket.send_text("An unexpected server error occurred in RAG App 2.")
        except Exception:
            pass
    finally:
        if websocket.client_state != websocket.client_state.DISCONNECTED:
            try:
                await websocket.close()
                print("RAG App 2: WebSocket connection closed.")
            except Exception as e:
                print(f"RAG App 2: Error closing WebSocket: {e}")

# Note on paths for /process-documents:
# The Dockerfile for this app (rag_app_client/Dockerfile) should use `WORKDIR /app`
# and `COPY . /app`. The docker-compose.yml mounts `./rag_app_client:/app`.
# Therefore, `documents_for_rag` folder from `./rag_app_client/documents_for_rag`
# will be available at `/app/documents_for_rag` in the container.
# The `CHROMA_DB_PATH` is `/app/chroma_data`.
# The updated path logic for `process_documents_endpoint` uses `os.path.dirname(os.path.abspath(__file__))`
# which resolves to `/app` in the container, so `os.path.join("/app", "documents_for_rag")` is correct.
