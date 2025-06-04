import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock # Import MagicMock for more flexible mocking

# Import the FastAPI app instance from rag_app_mcp_server.main
# To make this work, we need to ensure that the tests can find the rag_app_mcp_server module.
# This usually means the tests should be run from the repository root or PYTHONPATH needs to be adjusted.
# For now, assuming the execution context allows this import.
from rag_app_mcp_server.main import app

# Initialize the TestClient
client = TestClient(app)

# Dummy embedding and context for mocking
DUMMY_EMBEDDING = [0.1, 0.2, 0.3]
DUMMY_CONTEXT_CHUNKS = ["This is a context chunk.", "Another context chunk."]
MOCKED_LLM_RESPONSE = "Mocked LLM Response from MCP Server"

def test_websocket_connection():
    """Test basic WebSocket connection establishment and closure."""
    with client.websocket_connect("/ws/chat") as websocket:
        # If connect succeeds, the connection was accepted.
        # FastAPI TestClient's websocket_connect raises an exception on failure to connect/accept.
        print("WebSocket connected successfully.")
        # No specific assertion needed here for acceptance with TestClient,
        # as failure to connect/accept would raise an exception.
        websocket.close()
    print("WebSocket connection closed.")

@patch('rag_app_mcp_server.main.generate_response_from_context')
@patch('rag_app_mcp_server.main.query_collection_with_embedding')
@patch('rag_app_mcp_server.main.generate_query_embedding')
def test_websocket_chat_flow(
    mock_generate_query_embedding: MagicMock,
    mock_query_collection: MagicMock,
    mock_generate_response: MagicMock
):
    """Test the full chat flow over WebSocket with mocked RAG components."""
    # Configure mock return values
    mock_generate_query_embedding.return_value = [DUMMY_EMBEDDING]
    # query_collection_with_embedding returns a dict like {'documents': [['chunk1', 'chunk2']]}
    mock_query_collection.return_value = {'documents': [DUMMY_CONTEXT_CHUNKS]}
    mock_generate_response.return_value = MOCKED_LLM_RESPONSE

    with client.websocket_connect("/ws/chat") as websocket:
        test_message = "Hello Server, tell me about MCP."
        websocket.send_text(test_message)

        response = websocket.receive_text()
        assert response == MOCKED_LLM_RESPONSE

        # Verify that the mocked functions were called correctly
        mock_generate_query_embedding.assert_called_once_with([test_message])
        mock_query_collection.assert_called_once_with(query_embedding=DUMMY_EMBEDDING, n_results=3)
        mock_generate_response.assert_called_once_with(
            query=test_message,
            context_chunks=DUMMY_CONTEXT_CHUNKS,
            max_context_length=1500,
            max_new_tokens=150 # As per main.py in rag_app_mcp_server
        )
        websocket.close()

@patch('rag_app_mcp_server.main.llm_pipeline_global', None) # Mock llm_pipeline_global to be None
def test_websocket_chat_llm_unavailable():
    """Test WebSocket behavior when the LLM pipeline is unavailable."""
    with client.websocket_connect("/ws/chat") as websocket:
        # The connection should be accepted, but an error message about LLM
        # should be sent if we try to process a message.
        # In the current main.py, the check for llm_pipeline_global is done
        # at the beginning of the WebSocket session if chromadb/embedder are okay.

        # The current main.py for rag_app_mcp_server checks LLM pipeline at the start of the session.
        # It sends an error and closes if LLM is None.
        response = websocket.receive_text() # Should receive the error text
        assert "Error: Backend LLM pipeline not initialized." in response

        # Connection should be closed by the server after sending this error.
        # We can try to receive again, expecting a disconnect or specific close code.
        try:
            # Depending on TestClient's behavior for server-initiated close,
            # this might raise an exception or return a close message.
            # For now, we assume the error text is the primary check.
            further_response = websocket.receive_text(timeout=1) # short timeout
            # This part is tricky because the server closes the connection.
            # TestClient might raise an error here.
            # The main goal is to check the initial error message.
        except Exception as e: # pylint: disable=broad-except
            # This could be a WebSocketDisconnect exception or similar from TestClient
            print(f"Expected exception after server closes connection due to LLM unavailable: {e}")
            pass
            # No specific assertion on the close itself, the error message is key.
    # TestClient should handle the context exit cleanly.

@patch('rag_app_mcp_server.main.chromadb_collection', None) # Mock chromadb_collection to be None
def test_websocket_chat_chromadb_unavailable():
    """Test WebSocket behavior when ChromaDB is unavailable."""
    with client.websocket_connect("/ws/chat") as websocket:
        # Similar to LLM, ChromaDB is checked at the start of the session.
        response = websocket.receive_text()
        assert "Error: Backend RAG components (ChromaDB or Embedder) not initialized." in response

        # Connection should be closed by the server.
        try:
            websocket.receive_text(timeout=1)
        except Exception as e: # pylint: disable=broad-except
            print(f"Expected exception after server closes connection due to ChromaDB unavailable: {e}")
            pass
    # TestClient should handle the context exit cleanly.

# To run these tests:
# 1. Ensure your current directory is the root of the repository.
# 2. Make sure rag_app_mcp_server and its submodules are in PYTHONPATH.
#    One way is to run pytest with python -m pytest, or set PYTHONPATH=.
# 3. Command: pytest rag_app_mcp_server/tests/test_main_mcp_server.py
#
# Note on imports:
# The test assumes that 'rag_app_mcp_server.main' can be imported.
# If the Dockerfile for rag_app_mcp_server copies its content into /app,
# and tests are run in an environment where /app is the root for this app,
# then imports like 'from main import ...' would be used if tests are inside /app.
# However, these tests are in /app/tests.
# The current structure `from rag_app_mcp_server.main import app` assumes that
# the parent directory of `rag_app_mcp_server` is in PYTHONPATH,
# which is typical if you run `pytest` from the repository root.
# Example: If repo root is /my_project, and tests are in /my_project/rag_app_mcp_server/tests,
# running `pytest` from /my_project should make `rag_app_mcp_server` a top-level package.
#
# The `vector_store.py` in `rag_app_mcp_server` initializes ChromaDB client/collection globally on import.
# If these tests run, they might try to initialize a real ChromaDB instance if not properly managed.
# The `lifespan` manager in `main.py` also logs ChromaDB status.
# For these unit tests, ideally, the global `chromadb_collection` and `llm_pipeline_global`
# in `rag_app_mcp_server.main` should also be patched if they are accessed directly by `app`
# outside of the request flow, or if their state affects test setup.
# However, the WebSocket endpoint itself re-checks these globals.
# The @patch for `llm_pipeline_global` and `chromadb_collection` directly in `main`
# should control their state for the tests checking unavailability.
# For `test_websocket_chat_flow`, the RAG components are mocked, so their actual state doesn't matter.
# The `TestClient` initializes the app, including its lifespan manager.
# We might see lifespan log output during tests.
# To avoid issues with ChromaDB trying to write to disk during tests,
# ensure CHROMA_DB_PATH is either mocked or uses an in-memory/test-specific path
# if real instances were ever needed (they are not for these mocked tests).
# The current tests mock these components at the `main.py` level where they are imported and used by the endpoint.
