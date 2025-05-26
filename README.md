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

This section describes endpoints that offer a similar chat experience to `/chat` but are designated under the "MCP" (Manual Context Passthrough) path. Initially, MCP involved separate steps for context retrieval and response generation, but this has been consolidated into a single endpoint.

*   **`POST /mcp/chat`**:
    *   **Description**: Takes a user's message, performs context retrieval from the RAG system's vector store, and then generates a response using the Language Model (LLM) based on the user's query and the retrieved context. This endpoint mirrors the functionality of the main `/chat` endpoint.
    *   **Request Body**: `{ "message": "Your question for the MCP chat" }`
    *   **Response Body**: `{ "user_query": "Your question...", "retrieved_context_chunks": ["retrieved context chunk 1", "..."], "llm_response": "The LLM's generated answer" }`
