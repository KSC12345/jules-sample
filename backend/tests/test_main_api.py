import pytest
import pytest_asyncio
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock

# Ensure the app can be imported. This might require adjusting PYTHONPATH if tests are run from outside /app/backend
# For now, assume tests are run from /app/backend or PYTHONPATH is set.
# If main is not found, one solution is to add /app to sys.path in tests.
import sys
import os

# Add the parent directory (/app) to sys.path to allow imports like 'from main import app'
# This is often necessary when running tests from a subdirectory like /app/backend/tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import app # FastAPI app instance

# Base URL for the test client
BASE_URL = "http://test"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url=BASE_URL) as ac:
        yield ac

@pytest.mark.asyncio
async def test_read_root(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the RAG chatbot backend!"}

@pytest.mark.asyncio
@patch('main.process_docs_for_rag', new_callable=AsyncMock) # Mock the actual processing function
async def test_process_documents_endpoint_success(mock_process_docs: AsyncMock, client: AsyncClient):
    # Configure the mock to return a successful-like response
    mock_process_docs.return_value = {
        "status": "success",
        "processed_files": 1,
        "total_chunks_added": 10,
        "collection_total_items": 10
    }
    
    # Mock os.path.isdir to always return True for the default path
    with patch('os.path.isdir', return_value=True):
        response = await client.post("/process-documents", json={"directory": "backend/documents_for_rag"})
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Document processing initiated."
    assert json_response["details"]["status"] == "success"
    mock_process_docs.assert_called_once()

@pytest.mark.asyncio
async def test_process_documents_endpoint_dir_not_found(client: AsyncClient):
    # Mock os.path.isdir to return False
    with patch('os.path.isdir', return_value=False):
        response = await client.post("/process-documents", json={"directory": "non_existent_dir"})
    
    assert response.status_code == 400
    assert "Directory not found" in response.json()["detail"]


@pytest.mark.asyncio
@patch('main.generate_query_embedding')
@patch('main.query_collection_with_embedding')
@patch('main.generate_response_from_context')
@patch('main.llm_pipeline_global', new_callable=MagicMock) # Mock the pipeline object itself
@patch('main.chromadb_collection', new_callable=MagicMock) # Mock chromadb_collection
@patch('main.generate_query_embedding_module', new_callable=MagicMock) # Mock document_processor.generate_embeddings
async def test_chat_endpoint_success(
    mock_gen_query_embed_module: MagicMock, # This is for the check generate_query_embedding is None
    mock_chromadb_collection: MagicMock,
    mock_llm_pipeline: MagicMock,
    mock_generate_response: MagicMock,
    mock_query_chroma: MagicMock,
    mock_generate_query_embed: MagicMock,
    client: AsyncClient
):
    # Ensure mocked checks for initialization pass
    # The generate_query_embedding in main.py is an alias, 
    # so we mock what it points to: document_processor.generate_embeddings
    # For the check "if chromadb_collection is None or generate_query_embedding is None:"
    # We need to make sure these are not None.
    # The @patch for 'main.generate_query_embedding' handles the aliased name.
    # So, we need to ensure the check "if llm_pipeline_global is None" passes.
    # The patch for 'main.llm_pipeline_global' already ensures it's a MagicMock (not None).
    # Same for chromadb_collection.
    
    # Configure mock return values
    mock_generate_query_embed.return_value = [[0.1, 0.2, 0.3]] # Sample embedding
    mock_query_chroma.return_value = { # Sample response from ChromaDB query
        "documents": [["chunk1 text", "chunk2 text"]],
        "metadatas": [[{"source": "doc1"}, {"source": "doc2"}]],
        "distances": [[0.5, 0.6]]
    }
    mock_generate_response.return_value = "This is a mock LLM response."

    # Make the POST request
    chat_message = {"message": "Hello, tell me about something."}
    response = await client.post("/chat", json=chat_message)

    # Assertions
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["user_query"] == chat_message["message"]
    assert json_response["retrieved_context_chunks"] == ["chunk1 text", "chunk2 text"]
    assert json_response["llm_response"] == "This is a mock LLM response."

    # Verify that mocked functions were called
    mock_generate_query_embed.assert_called_once_with([chat_message["message"]])
    mock_query_chroma.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], n_results=3)
    mock_generate_response.assert_called_once_with(
        query=chat_message["message"],
        context_chunks=["chunk1 text", "chunk2 text"],
        max_context_length=1500,
        max_new_tokens=100
    )

@pytest.mark.asyncio
@patch('main.llm_pipeline_global', None) # Simulate LLM pipeline not loaded
@patch('main.chromadb_collection', new_callable=MagicMock)
@patch('main.generate_query_embedding', new_callable=MagicMock)
async def test_chat_endpoint_llm_not_loaded(
    mock_generate_query_embed: MagicMock,
    mock_chromadb_collection: MagicMock,
    client: AsyncClient
):
    # We've mocked llm_pipeline_global to be None
    chat_message = {"message": "Test message"}
    response = await client.post("/chat", json=chat_message)
    
    assert response.status_code == 500
    assert "LLM generation pipeline is not available" in response.json()["detail"]

@pytest.mark.asyncio
@patch('main.chromadb_collection', None) # Simulate ChromaDB not initialized
@patch('main.generate_query_embedding', new_callable=MagicMock) # Mock generate_query_embedding
async def test_chat_endpoint_chromadb_not_loaded(
    mock_generate_query_embedding: MagicMock,
    client: AsyncClient):

    chat_message = {"message": "Test message"}
    response = await client.post("/chat", json=chat_message)

    assert response.status_code == 500
    assert "ChromaDB or SentenceTransformer model not initialized" in response.json()["detail"]


# --- Tests for /mcp/chat ---

@pytest.mark.asyncio
@patch('main.generate_query_embedding')
@patch('main.query_collection_with_embedding')
@patch('main.generate_response_from_context')
@patch('main.llm_pipeline_global', new_callable=MagicMock)
@patch('main.chromadb_collection', new_callable=MagicMock)
async def test_mcp_chat_success(
    mock_chromadb_collection: MagicMock,
    mock_llm_pipeline: MagicMock,
    mock_generate_response: MagicMock,
    mock_query_chroma: MagicMock,
    mock_generate_query_embed: MagicMock,
    client: AsyncClient
):
    # Configure mock return values
    mock_generate_query_embed.return_value = [[0.1, 0.2, 0.3]] # Sample embedding
    mock_query_chroma.return_value = {
        "documents": [["mcp_chunk1 text", "mcp_chunk2 text"]],
        "metadatas": [[{"source": "doc1"}, {"source": "doc2"}]],
        "distances": [[0.5, 0.6]]
    }
    mock_generate_response.return_value = "This is a mock MCP LLM response."

    # Make the POST request
    mcp_chat_message = {"message": "Hello, MCP!"}
    response = await client.post("/mcp/chat", json=mcp_chat_message)

    # Assertions
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["user_query"] == mcp_chat_message["message"]
    assert json_response["retrieved_context_chunks"] == ["mcp_chunk1 text", "mcp_chunk2 text"]
    assert json_response["llm_response"] == "This is a mock MCP LLM response."

    # Verify that mocked functions were called
    mock_generate_query_embed.assert_called_once_with([mcp_chat_message["message"]])
    mock_query_chroma.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], n_results=3)
    mock_generate_response.assert_called_once_with(
        query=mcp_chat_message["message"],
        context_chunks=["mcp_chunk1 text", "mcp_chunk2 text"],
        max_context_length=1500,
        max_new_tokens=100
    )

@pytest.mark.asyncio
@patch('main.generate_query_embedding', None) # Simulate embedding model not loaded
@patch('main.chromadb_collection', new_callable=MagicMock) # Mock chromadb_collection to be present
@patch('main.llm_pipeline_global', new_callable=MagicMock) # Mock llm_pipeline_global to be present
async def test_mcp_chat_embedding_model_not_loaded(
    mock_llm_pipeline: MagicMock,
    mock_chromadb_collection: MagicMock,
    client: AsyncClient
):
    mcp_chat_message = {"message": "Test message for embedding failure"}
    response = await client.post("/mcp/chat", json=mcp_chat_message)
    
    assert response.status_code == 500
    # The error message in main.py for /mcp/chat is "ChromaDB or SentenceTransformer model not initialized. Cannot process MCP chat."
    assert "ChromaDB or SentenceTransformer model not initialized" in response.json()["detail"]
    assert "Cannot process MCP chat" in response.json()["detail"]


@pytest.mark.asyncio
@patch('main.chromadb_collection', None) # Simulate ChromaDB not initialized
@patch('main.generate_query_embedding', new_callable=MagicMock) # Mock generate_query_embedding to be present
@patch('main.llm_pipeline_global', new_callable=MagicMock) # Mock llm_pipeline_global to be present
async def test_mcp_chat_chromadb_not_loaded(
    mock_llm_pipeline: MagicMock,
    mock_generate_query_embedding: MagicMock,
    client: AsyncClient
):
    mcp_chat_message = {"message": "Test message for chromadb failure"}
    response = await client.post("/mcp/chat", json=mcp_chat_message)

    assert response.status_code == 500
    # The error message in main.py for /mcp/chat is "ChromaDB or SentenceTransformer model not initialized. Cannot process MCP chat."
    assert "ChromaDB or SentenceTransformer model not initialized" in response.json()["detail"]
    assert "Cannot process MCP chat" in response.json()["detail"]


@pytest.mark.asyncio
@patch('main.llm_pipeline_global', None) # Simulate LLM pipeline not loaded
@patch('main.chromadb_collection', new_callable=MagicMock) # Mock chromadb_collection to be present
@patch('main.generate_query_embedding', new_callable=MagicMock) # Mock generate_query_embedding to be present
async def test_mcp_chat_llm_not_loaded(
    mock_generate_query_embedding: MagicMock,
    mock_chromadb_collection: MagicMock,
    client: AsyncClient
):
    # Configure generate_query_embedding to return a valid embedding to pass that stage
    mock_generate_query_embedding.return_value = [[0.1, 0.2, 0.3]]
    
    mcp_chat_message = {"message": "Test message for LLM failure"}
    response = await client.post("/mcp/chat", json=mcp_chat_message)
    
    assert response.status_code == 500
    # The error message in main.py for /mcp/chat is "LLM generation pipeline is not available. Cannot generate MCP response."
    assert "LLM generation pipeline is not available" in response.json()["detail"]
    assert "Cannot generate MCP response" in response.json()["detail"]


@pytest.mark.asyncio
@patch('main.generate_query_embedding')
@patch('main.query_collection_with_embedding')
@patch('main.generate_response_from_context')
@patch('main.llm_pipeline_global', new_callable=MagicMock)
@patch('main.chromadb_collection', new_callable=MagicMock)
async def test_mcp_chat_no_chunks_found(
    mock_chromadb_collection: MagicMock,
    mock_llm_pipeline: MagicMock,
    mock_generate_response: MagicMock,
    mock_query_chroma: MagicMock,
    mock_generate_query_embed: MagicMock,
    client: AsyncClient
):
    # Configure mock return values
    mock_generate_query_embed.return_value = [[0.1, 0.2, 0.3]]
    mock_query_chroma.return_value = {"documents": [[]]} # No documents found
    mock_generate_response.return_value = "This is a mock MCP LLM response even with no chunks."

    mcp_chat_message = {"message": "Query for no chunks"}
    response = await client.post("/mcp/chat", json=mcp_chat_message)

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["user_query"] == mcp_chat_message["message"]
    assert json_response["retrieved_context_chunks"] == []
    assert json_response["llm_response"] == "This is a mock MCP LLM response even with no chunks."

    mock_generate_query_embed.assert_called_once_with([mcp_chat_message["message"]])
    mock_query_chroma.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], n_results=3)
    mock_generate_response.assert_called_once_with(
        query=mcp_chat_message["message"],
        context_chunks=[], # Empty list for context
        max_context_length=1500,
        max_new_tokens=100
    )

# To run these tests, navigate to the /app/backend directory and run:
# python -m pytest
# Ensure PYTHONPATH includes /app if running from /app/backend/tests directly or if imports fail.
# The sys.path.insert above should help when running with `python -m pytest` from `/app/backend`.
