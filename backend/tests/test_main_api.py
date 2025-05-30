import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch # Keep patch if needed for other things, though monkeypatch is often for config
import importlib # For reloading modules after monkeypatching config

from backend.main import app # FastAPI app instance
# Assuming figma_retriever and llm_generator will use their dummy data
# when API keys are not set (monkeypatched to None in backend.config).

BASE_URL = "http://test" # Base URL for the test client

@pytest_asyncio.fixture
async def client():
    """Provides an HTTPX AsyncClient for making requests to the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
        yield ac

@pytest.mark.asyncio
async def test_read_root(client: AsyncClient):
    """Test the root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the RAG chatbot backend!"}

@pytest.mark.asyncio
async def test_generate_react_from_figma_success_dummy_data(client: AsyncClient, monkeypatch):
    """
    Integration test for /generate-react-from-figma using dummy data fallbacks.
    Ensures the endpoint orchestrates the RAG process correctly when external APIs are "off".
    """
    # 1. Ensure API keys are None so dummy data is used by sub-modules
    monkeypatch.setattr('backend.config.FIGMA_API_KEY', None)
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', None)

    # 2. Reload modules that initialize clients/settings based on config at import time
    # This is crucial for the monkeypatched config to take effect in those modules.
    import backend.figma_retriever
    import backend.llm_generator
    import backend.react_generator_service # This service imports the above two
    import backend.vector_store # Ensure vector store also uses potentially fresh config if needed
                                # (though its DB path is usually set by test_vector_store.py for its own tests)
                                # For this API test, it will use its default dev DB path unless that's also
                                # monkeypatched here to a test-specific temporary DB.
                                # For now, assume it uses the default dev DB which is fine for this integration test flow.

    importlib.reload(backend.figma_retriever)
    importlib.reload(backend.llm_generator)
    importlib.reload(backend.vector_store) # Reload to ensure it's using consistent config state if necessary
    importlib.reload(backend.react_generator_service)


    # 3. Populate the vector store with dummy React components
    # Call the /process-components endpoint to achieve this.
    # This makes the test more of an integration test.
    print("Populating vector store by calling /process-components...")
    process_response = await client.post("/process-components")
    assert process_response.status_code == 200
    # Check if components were added (usually 2: Button.jsx, Card.jsx from default dir)
    # The count might vary if previous tests left items and DB is not reset per API test run.
    # For this test, we mostly care that it ran and likely populated *something*.
    print(f"Vector store population response: {process_response.json()}")
    assert process_response.json().get("processed_count", 0) > 0, "Vector store should have processed some components."


    # 4. Prepare and make the request to /generate-react-from-figma
    payload = {
        "figma_file_key": "dummy_test_file_key", # Will use dummy figma_retriever
        "node_ids": ["1:1", "1:2"] # Node IDs known to be in figma_retriever's DUMMY_NODES_INFO_RESPONSE
    }
    print(f"Requesting /generate-react-from-figma with payload: {payload}")
    response = await client.post("/generate-react-from-figma", json=payload)

    # 5. Assertions on the response
    assert response.status_code == 200
    json_response = response.json()
    
    assert "results" in json_response
    results_list = json_response["results"]
    assert isinstance(results_list, list)
    assert len(results_list) == len(payload["node_ids"])

    for i, item in enumerate(results_list):
        node_id = payload["node_ids"][i]
        assert item["figma_node_id"] == node_id
        assert "figma_component_name" in item
        assert "generated_component_name" in item
        assert "generated_react_code" in item
        
        # Check that dummy Figma data was used
        if node_id == "1:1":
            assert item["figma_component_name"] == "Frame1" # From DUMMY_NODES_INFO_RESPONSE
            assert item["generated_component_name"] == "Frame1"
        elif node_id == "1:2":
            assert item["figma_component_name"] == "Rectangle1" # From DUMMY_NODES_INFO_RESPONSE
            assert item["generated_component_name"] == "Rectangle1"

        # Check that dummy LLM response was used
        assert "DummyFigmaComponent_" in item["generated_react_code"]
        assert f"// Component Name: {item['generated_component_name']}" in item["generated_react_code"]
        
        # Check that vector store retrieval happened (source should be one of our dummy components)
        assert item["retrieved_context_source"] in ["Button.jsx", "Card.jsx"], \
            f"Unexpected retrieved_context_source: {item['retrieved_context_source']}"
        print(f"Processed result for node {node_id} looks OK.")

@pytest.mark.asyncio
async def test_generate_react_from_figma_invalid_input(client: AsyncClient):
    """Test /generate-react-from-figma with various invalid inputs."""
    
    invalid_payloads = [
        {}, # Empty payload
        {"figma_file_key": "key_only"}, # Missing node_ids
        {"node_ids": ["1:1"]}, # Missing figma_file_key
        {"figma_file_key": "key", "node_ids": "not_a_list"}, # node_ids wrong type
            # {"figma_file_key": "key", "node_ids": []}, # Removed: Empty list is valid, returns 200. Tested separately.
    ]
    
    for payload in invalid_payloads:
        print(f"Testing invalid payload: {payload}")
        response = await client.post("/generate-react-from-figma", json=payload)
        # FastAPI returns 422 for Pydantic validation errors
        assert response.status_code == 422, f"Failed for payload: {payload}. Response: {response.text}"

    # Test with node_ids list being empty - specific check as per Pydantic model (Field ... means required)
    # Actually, Pydantic `List[str] = Field(...)` means the list itself is required, but can be empty.
    # The endpoint logic might handle empty node_ids list gracefully (e.g. return empty results).
    # Let's test this specific case for a 200 response with empty results.
    print("Testing with empty node_ids list specifically...")
    empty_node_ids_payload = {"figma_file_key": "key", "node_ids": []}
    response_empty_nodes = await client.post("/generate-react-from-figma", json=empty_node_ids_payload)
    assert response_empty_nodes.status_code == 200 # Expecting it to process an empty list of nodes successfully
    json_data_empty_nodes = response_empty_nodes.json()
    assert json_data_empty_nodes == {"results": []}
    print("Empty node_ids list handled correctly.")

# Keep other commented-out tests below if they are planned for future work,
# or remove them if they are definitively obsolete.
# For now, this file focuses on / and /generate-react-from-figma.
pass
