import grpc
from concurrent import futures
import os
import sys

# Add backend directory to sys.path to allow direct imports for shared modules
# This might be needed if running the gRPC server independently or in certain environments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mcp_pb2
import mcp_pb2_grpc

# Import functionalities from other backend modules
try:
    from vector_store import (
        process_and_embed_documents as process_docs_for_rag_internal,
        query_collection_with_embedding,
        collection as chromadb_collection,
    )
    from document_processor import generate_embeddings as generate_query_embedding
    from llm_generator import generate_response_from_context, generator_pipeline as llm_pipeline_global
except ImportError as e:
    print(f"Error importing shared modules in mcp_server.py: {e}")
    # Set them to None so the server can start but RPCs will report errors
    process_docs_for_rag_internal = None
    query_collection_with_embedding = None
    chromadb_collection = None
    generate_query_embedding = None
    generate_response_from_context = None
    llm_pipeline_global = None

class MCPServiceServicer(mcp_pb2_grpc.MCPServiceServicer):
    def Chat(self, request: mcp_pb2.MCPChatMessage, context):
        print(f"MCP Server: Received Chat request: {request.message[:50]}...")
        user_message = request.message

        if not all([chromadb_collection, generate_query_embedding, llm_pipeline_global, query_collection_with_embedding, generate_response_from_context]):
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("A critical component (DB, embedding model, or LLM) is not available.")
            return mcp_pb2.MCPChatResponse()

        try:
            query_embedding_list = generate_query_embedding([user_message])
            if not query_embedding_list or not query_embedding_list[0]:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to generate embedding for the user message.")
                return mcp_pb2.MCPChatResponse()
            
            query_embedding = query_embedding_list[0]

            retrieved_info = query_collection_with_embedding(
                query_embedding=query_embedding,
                n_results=3
            )

            retrieved_doc_chunks = []
            if retrieved_info and retrieved_info.get('documents') and retrieved_info['documents'][0]:
                retrieved_doc_chunks = retrieved_info['documents'][0]
            
            print(f"MCP Server: Retrieved {len(retrieved_doc_chunks)} chunks for query '{user_message[:30]}...'.")

            llm_response_text = generate_response_from_context(
                query=user_message,
                context_chunks=retrieved_doc_chunks,
                max_context_length=1500,
                max_new_tokens=100
            )

            return mcp_pb2.MCPChatResponse(
                user_query=user_message,
                retrieved_context_chunks=retrieved_doc_chunks,
                llm_response=llm_response_text
            )
        except Exception as e:
            print(f"MCP Server: Error during Chat RPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An unexpected error occurred: {str(e)}")
            return mcp_pb2.MCPChatResponse()

    def ProcessDocuments(self, request: mcp_pb2.MCPProcessRequest, context):
        print(f"MCP Server: Received ProcessDocuments request for directory: {request.directory}")
        docs_dir = request.directory

        if not all([chromadb_collection, generate_query_embedding, process_docs_for_rag_internal]):
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("A critical component (DB or embedding model) is not available for document processing.")
            return mcp_pb2.MCPProcessResponse(
                message="Error: Critical component not available.",
                status="error"
            )

        # Ensure the path is absolute or resolve it relative to /app (common in Docker)
        # This logic mirrors main.py
        if not os.path.isabs(docs_dir):
            # Assuming /app is the root for relative paths if not absolute
            # This path might need adjustment depending on where the gRPC server is run from.
            # For consistency with FastAPI app, let's assume /app is the intended base.
            # However, if mcp_server.py is run directly from backend/, CWD is backend/.
            # If this server is always started via main.py in the root, then /app is fine.
            # For now, let's use a path that might be more robust if run from backend/
            # but ideally, this should be harmonized.
             base_path = os.environ.get("APP_BASE_PATH", "/app") # Default to /app
             docs_dir_abs = os.path.join(base_path, docs_dir)
             # A simpler approach if server is always run from project root:
             # docs_dir_abs = os.path.abspath(docs_dir)

        # Correct path resolution: if docs_dir is relative, it should be relative to the project root.
        # The original main.py assumes /app is the root when running in Docker.
        # If running locally, this path needs careful handling.
        # For the MCP server, let's assume 'docs_dir' is either absolute
        # or relative to the project root ('/app' in Docker, or repo root locally).
        
        # Simplified path logic for now, assuming `docs_dir` is relative to project root or absolute.
        # If relative, make it absolute from current working directory (which should be project root)
        if not os.path.isabs(docs_dir):
             # This assumes the server is started from the project root.
             # If it's started from `backend/`, this will be backend/backend/documents_for_rag
             # This needs to be robust. A common pattern is to define a BASE_DIR at the project root.
             # For now, let's try to make it work assuming it's relative to /app or similar.
             # The logic in main.py uses os.path.join("/app", docs_dir)
             # We should aim for similar behavior.
            if os.getenv("RUNNING_IN_DOCKER"): # A way to signal Docker environment
                 docs_dir_abs = os.path.join("/app", docs_dir)
            else:
                 # Assuming script is in backend/, and project root is parent of backend/
                 project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 docs_dir_abs = os.path.join(project_root, docs_dir)

        else:
            docs_dir_abs = docs_dir
        
        print(f"MCP Server: Absolute path for document processing: {docs_dir_abs}")


        if not os.path.isdir(docs_dir_abs):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Directory not found: {docs_dir_abs}")
            return mcp_pb2.MCPProcessResponse(
                message=f"Error: Directory not found: {docs_dir_abs}",
                status="error"
            )

        try:
            # Call the internal function, ensuring it uses the absolute path
            result = process_docs_for_rag_internal(documents_dir=docs_dir_abs)
            
            if result.get("status") == "error":
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(result.get("message", "Unknown error during document processing."))
                return mcp_pb2.MCPProcessResponse(
                    message=result.get("message", "Unknown error"),
                    status="error"
                )
            
            return mcp_pb2.MCPProcessResponse(
                message=f"Document processing initiated for {docs_dir_abs}.", # Corrected message
                status=result.get("status", "success"),
                processed_files=result.get("processed_files", 0),
                total_chunks_added=result.get("total_chunks_added", 0),
                collection_total_items=result.get("collection_total_items", 0)
            )
        except Exception as e:
            print(f"MCP Server: Error during ProcessDocuments RPC: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An unexpected error occurred during document processing: {str(e)}")
            return mcp_pb2.MCPProcessResponse(
                message=f"An unexpected error occurred: {str(e)}",
                status="error"
            )

def serve():
    # Check if critical components failed to load during import
    if not all([process_docs_for_rag_internal, query_collection_with_embedding, chromadb_collection, 
                generate_query_embedding, generate_response_from_context, llm_pipeline_global]):
        print("MCP Server: Critical components (DB, embedding model, or LLM) failed to load.")
        print("MCP server cannot start properly. Please check logs for import errors in shared modules.")
        # Optionally, exit or prevent server from starting if critical components are missing
        # For now, it will start but RPCs will fail with UNAVAILABLE status.
        # return # Uncomment to prevent server from starting

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mcp_pb2_grpc.add_MCPServiceServicer_to_server(MCPServiceServicer(), server)
    
    port = "50051"
    server.add_insecure_port(f'[::]:{port}')
    
    print(f"MCP Server starting on port {port}...")
    try:
        server.start()
        print(f"MCP Server started successfully on port {port}.")
        server.wait_for_termination()
    except Exception as e:
        print(f"MCP Server: Failed to start or run: {e}")
    finally:
        server.stop(0) # Ensure server is stopped if loop is broken or error occurs
        print("MCP Server stopped.")

if __name__ == '__main__':
    # This allows running the gRPC server directly for testing
    print("Attempting to start standalone MCP gRPC server...")
    # Perform basic checks for shared module components
    if chromadb_collection is None:
        print("WARNING (standalone mcp_server): ChromaDB collection is None. Document processing and chat might fail.")
    if llm_pipeline_global is None:
        print("WARNING (standalone mcp_server): LLM pipeline is None. Chat functionality will likely fail.")
    if generate_query_embedding is None:
         print("WARNING (standalone mcp_server): Embedding generation function is None. Most functionalities will fail.")
    
    serve()
