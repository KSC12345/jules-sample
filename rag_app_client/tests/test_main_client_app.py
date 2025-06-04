import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import websockets # For websockets.exceptions

# Import the FastAPI app instance and other necessary components from rag_app_client.main
from rag_app_client.main import app, should_forward_to_mcp, MCP_SERVER_URL # Import MCP_SERVER_URL for verification

# Initialize the TestClient
client = TestClient(app)

# Dummy data for mocking local RAG
DUMMY_LOCAL_EMBEDDING = [0.4, 0.5, 0.6]
DUMMY_LOCAL_CONTEXT_CHUNKS = ["Local context chunk 1.", "Local context chunk 2."]
MOCKED_LOCAL_LLM_RESPONSE = "Mocked Local LLM Response for Client App"
MOCKED_MCP_RESPONSE = "Mocked MCP Response via Forwarding"

def test_websocket_connection_client_app():
    """Test basic WebSocket connection to RAG App 2."""
    with client.websocket_connect("/ws/chat") as websocket:
        assert websocket is not None # Connection accepted
        websocket.close()

def test_should_forward_to_mcp_logic():
    """Test the should_forward_to_mcp decision function directly."""
    assert should_forward_to_mcp("FORWARD_TO_MCP: Hello") == True
    assert should_forward_to_mcp("forward_to_mcp: Hello") == True # Case-insensitive check due to .upper()
    assert should_forward_to_mcp("Hello FORWARD_TO_MCP") == True
    assert should_forward_to_mcp("Hello regular message") == False
    assert should_forward_to_mcp("") == False

@patch('rag_app_client.main.generate_response_from_context')
@patch('rag_app_client.main.query_collection_with_embedding')
@patch('rag_app_client.main.generate_query_embedding')
def test_chat_local_processing(
    mock_generate_query_embedding: MagicMock,
    mock_query_collection: MagicMock,
    mock_generate_response: MagicMock
):
    """Test the local RAG processing path in RAG App 2."""
    mock_generate_query_embedding.return_value = [DUMMY_LOCAL_EMBEDDING]
    mock_query_collection.return_value = {'documents': [DUMMY_LOCAL_CONTEXT_CHUNKS]}
    mock_generate_response.return_value = MOCKED_LOCAL_LLM_RESPONSE

    with client.websocket_connect("/ws/chat") as websocket:
        test_message = "Tell me about local weather" # Does not contain FORWARD_TO_MCP
        websocket.send_text(test_message)
        response = websocket.receive_text()

        assert response == f"Locally Processed: {MOCKED_LOCAL_LLM_RESPONSE}"
        mock_generate_query_embedding.assert_called_once_with([test_message])
        mock_query_collection.assert_called_once_with(query_embedding=DUMMY_LOCAL_EMBEDDING, n_results=3)
        mock_generate_response.assert_called_once_with(
            query=test_message,
            context_chunks=DUMMY_LOCAL_CONTEXT_CHUNKS,
            max_context_length=1500,
            max_new_tokens=150
        )
        websocket.close()

@patch('rag_app_client.main.websockets.connect', new_callable=AsyncMock)
def test_chat_forwarding_successful(mock_websockets_connect: AsyncMock):
    """Test successful forwarding of a query to the MCP server."""
    # Configure the mock for websockets.connect()
    mock_mcp_ws_client = AsyncMock() # This is the object returned by `async with websockets.connect(...) as mcp_ws:`
    mock_mcp_ws_client.send = AsyncMock()
    mock_mcp_ws_client.recv = AsyncMock(return_value=MOCKED_MCP_RESPONSE)

    # __aenter__ is what `async with` calls. It should return the mock client.
    mock_websockets_connect.return_value.__aenter__.return_value = mock_mcp_ws_client

    with client.websocket_connect("/ws/chat") as websocket:
        forward_message = "FORWARD_TO_MCP: Tell me about quantum physics"
        websocket.send_text(forward_message)
        response = websocket.receive_text()

        assert response == f"Forwarded to MCP, Response: {MOCKED_MCP_RESPONSE}"

        # Verify websockets.connect was called with the correct URL
        # MCP_SERVER_URL is imported from main, so it reflects the actual configuration (env var or default)
        mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)

        # Verify interaction with the mocked MCP WebSocket
        mock_mcp_ws_client.send.assert_called_once_with(forward_message)
        mock_mcp_ws_client.recv.assert_called_once()
        websocket.close()

@patch('rag_app_client.main.websockets.connect', new_callable=AsyncMock)
def test_chat_forwarding_mcp_connection_refused_error(mock_websockets_connect: AsyncMock):
    """Test forwarding when MCP server connection is refused."""
    mock_websockets_connect.side_effect = ConnectionRefusedError("Test connection refused")

    with client.websocket_connect("/ws/chat") as websocket:
        forward_message = "FORWARD_TO_MCP: This will fail to connect"
        websocket.send_text(forward_message)
        response = websocket.receive_text()

        assert "Error: RAG App 2: Could not connect to MCP server" in response
        assert "Connection refused" in response
        mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)
        websocket.close()

@patch('rag_app_client.main.websockets.connect', new_callable=AsyncMock)
def test_chat_forwarding_mcp_connection_closed_error(mock_websockets_connect: AsyncMock):
    """Test forwarding when MCP server connection closes unexpectedly."""
    # Simulate ConnectionClosedError, e.g. websockets.exceptions.ConnectionClosedError
    mock_websockets_connect.side_effect = websockets.exceptions.ConnectionClosedError(None, None, "Test connection closed")


    with client.websocket_connect("/ws/chat") as websocket:
        forward_message = "FORWARD_TO_MCP: This will also fail to connect"
        websocket.send_text(forward_message)
        response = websocket.receive_text()

        assert "Error: RAG App 2: MCP server connection closed." in response
        mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)
        websocket.close()


@patch('rag_app_client.main.websockets.connect', new_callable=AsyncMock)
def test_chat_forwarding_mcp_communication_error_on_recv(mock_websockets_connect: AsyncMock):
    """Test error during recv() from MCP server after successful connection."""
    mock_mcp_ws_client = AsyncMock()
    mock_mcp_ws_client.send = AsyncMock()
    # Simulate an error during recv
    mock_mcp_ws_client.recv = AsyncMock(side_effect=websockets.exceptions.WebSocketException("Test recv error"))

    mock_websockets_connect.return_value.__aenter__.return_value = mock_mcp_ws_client

    with client.websocket_connect("/ws/chat") as websocket:
        forward_message = "FORWARD_TO_MCP: This will error on recv"
        websocket.send_text(forward_message)
        response = websocket.receive_text()

        assert "Error: RAG App 2: Error during forwarding to MCP: Test recv error" in response
        mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)
        mock_mcp_ws_client.send.assert_called_once_with(forward_message)
        mock_mcp_ws_client.recv.assert_called_once() # Ensure recv was attempted
        websocket.close()

@patch('rag_app_client.main.websockets.connect', new_callable=AsyncMock)
def test_chat_forwarding_mcp_communication_error_on_send(mock_websockets_connect: AsyncMock):
    """Test error during send() to MCP server after successful connection."""
    mock_mcp_ws_client = AsyncMock()
    # Simulate an error during send
    mock_mcp_ws_client.send = AsyncMock(side_effect=websockets.exceptions.WebSocketException("Test send error"))
    mock_mcp_ws_client.recv = AsyncMock() # recv should not be called if send fails

    mock_websockets_connect.return_value.__aenter__.return_value = mock_mcp_ws_client

    with client.websocket_connect("/ws/chat") as websocket:
        forward_message = "FORWARD_TO_MCP: This will error on send"
        websocket.send_text(forward_message)
        response = websocket.receive_text()

        assert "Error: RAG App 2: Error during forwarding to MCP: Test send error" in response
        mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)
        mock_mcp_ws_client.send.assert_called_once_with(forward_message) # Ensure send was attempted
        mock_mcp_ws_client.recv.assert_not_called() # Recv should not be attempted if send fails
        websocket.close()

# Notes for running:
# - Ensure pytest and pytest-asyncio are installed (pytest-asyncio for AsyncMock if it were driving async tests).
#   TestClient itself provides a sync interface to FastAPI's async app.
# - Run from the repository root so that `from rag_app_client.main import app` works.
# - Command: pytest rag_app_client/tests/test_main_client_app.py
# - The MCP_SERVER_URL is imported from main.py, which means it will use os.getenv.
#   For tests, it will likely fall back to "ws://localhost:8001/ws/chat" unless the env var is set during test execution.
#   This is fine as `websockets.connect` is mocked, so no actual connection is made.
#   The assertion `mock_websockets_connect.assert_called_once_with(MCP_SERVER_URL)` correctly checks
#   that the code attempts to connect to whatever MCP_SERVER_URL is resolved to.
