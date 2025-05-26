# jules-sample

uvicorn main:app --host 0.0.0.0 --port 8000

## API Endpoints

This project provides a backend API for a RAG (Retrieval Augmented Generation) chatbot.

### Standard Endpoints

*   `GET /`: Welcome message.
*   `POST /process-documents`: Triggers processing of documents in the specified directory (default: `backend/documents_for_rag`) to load them into the RAG system.
    *   Request body: `{ "directory": "path/to/your/documents" }` (optional, defaults to `backend/documents_for_rag`)
    *   Response: Confirmation of processing initiation or error.
*   `POST /chat`: Main chat endpoint. Takes a user message, retrieves context, and generates an LLM response.
    *   Request body: `{ "message": "Your question here" }`
    *   Response: `{ "user_query": "...", "retrieved_context_chunks": [...], "llm_response": "..." }`

### MCP API Endpoints (Manual Context Passthrough)

These endpoints allow for more granular control over the RAG process, separating context retrieval and response generation.

*   **`POST /mcp/context`**:
    *   **Description**: Takes a user message and retrieves relevant context chunks from the RAG system's vector store.
    *   **Request Body**: `{ "message": "Your query to find context for" }`
    *   **Response Body**: `{ "user_query": "Your query...", "retrieved_context_chunks": ["context chunk 1", "context chunk 2", ...] }`

*   **`POST /mcp/response`**:
    *   **Description**: Takes a user query and a list of context chunks, then generates a response using the RAG system's Language Model (LLM).
    *   **Request Body**: `{ "user_query": "Your original query", "context_chunks": ["previously retrieved context chunk 1", "chunk 2"] }`
    *   **Response Body**: `{ "llm_response": "The LLM's generated answer" }`
