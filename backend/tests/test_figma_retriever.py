import pytest
import requests
from unittest.mock import patch, MagicMock

# Assuming figma_retriever.py is in backend directory, and tests are run with /app as root or PYTHONPATH correctly set.
from backend.figma_retriever import get_figma_file_info, get_figma_nodes_info, FIGMA_API_KEY, DUMMY_FILE_INFO_RESPONSE, DUMMY_NODES_INFO_RESPONSE, HTTPException, PlaceholderHTTPException

# Use PlaceholderHTTPException for tests if actual HTTPException (from FastAPI) isn't available
# This makes tests runnable even if FastAPI isn't in the test environment for some reason,
# though typically it would be.
EffectiveHTTPException = HTTPException if HTTPException != PlaceholderHTTPException else PlaceholderHTTPException


@pytest.fixture
def mock_requests_get():
    """Mocks requests.get method."""
    with patch('requests.get') as mock_get:
        yield mock_get

# --- Tests for get_figma_file_info ---

def test_get_file_info_api_key_missing(monkeypatch, mock_requests_get):
    """Test get_figma_file_info when FIGMA_API_KEY is not set."""
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', None)

    file_key = "test_file_key"
    result = get_figma_file_info(file_key)

    assert result == DUMMY_FILE_INFO_RESPONSE
    mock_requests_get.assert_not_called()

def test_get_file_info_success(monkeypatch, mock_requests_get):
    """Test get_figma_file_info with a successful API response."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key"
    expected_response_json = {"name": "Live Figma File", "document": {"id": "0:1", "type": "DOCUMENT"}}

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = expected_response_json
    mock_requests_get.return_value = mock_response

    result = get_figma_file_info(file_key)

    assert result == expected_response_json
    mock_requests_get.assert_called_once_with(
        f"https://api.figma.com/v1/files/{file_key}",
        headers={"X-Figma-Token": api_key}
    )

def test_get_file_info_api_error(monkeypatch, mock_requests_get):
    """Test get_figma_file_info with a Figma API error response."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key_error"

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 403 # Forbidden, as an example
    mock_response.text = "Forbidden access"
    mock_requests_get.return_value = mock_response

    with pytest.raises(EffectiveHTTPException) as exc_info:
        get_figma_file_info(file_key)

    assert exc_info.value.status_code == 403
    assert "Forbidden access" in exc_info.value.detail
    mock_requests_get.assert_called_once()

def test_get_file_info_request_exception(monkeypatch, mock_requests_get):
    """Test get_figma_file_info with a requests library exception."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key_req_exception"
    mock_requests_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

    with pytest.raises(EffectiveHTTPException) as exc_info:
        get_figma_file_info(file_key)

    assert exc_info.value.status_code == 503 # Service Unavailable
    assert "Connection failed" in exc_info.value.detail
    mock_requests_get.assert_called_once()


# --- Tests for get_figma_nodes_info ---

def test_get_nodes_info_api_key_missing(monkeypatch, mock_requests_get):
    """Test get_figma_nodes_info when FIGMA_API_KEY is not set."""
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', None)

    file_key = "test_file_key"
    node_ids = ["1:1", "1:2"]
    # The dummy response generation for nodes is a bit complex,
    # it tries to filter DUMMY_NODES_INFO_RESPONSE or create placeholders.
    # Let's check against the expected structure for these specific IDs.
    expected_dummy_nodes = {
        node_id: DUMMY_NODES_INFO_RESPONSE["nodes"][node_id]
        for node_id in node_ids if node_id in DUMMY_NODES_INFO_RESPONSE["nodes"]
    }
    expected_result = {"nodes": expected_dummy_nodes, "name": DUMMY_NODES_INFO_RESPONSE["name"]}

    result = get_figma_nodes_info(file_key, node_ids)

    assert result == expected_result
    mock_requests_get.assert_not_called()

def test_get_nodes_info_success(monkeypatch, mock_requests_get):
    """Test get_figma_nodes_info with a successful API response."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key"
    node_ids = ["1:1", "1:2"]
    ids_param = ",".join(node_ids)
    expected_response_json = {"nodes": {"1:1": {"document": {}}, "1:2": {"document": {}}}}

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = expected_response_json
    mock_requests_get.return_value = mock_response

    result = get_figma_nodes_info(file_key, node_ids)

    assert result == expected_response_json
    mock_requests_get.assert_called_once_with(
        f"https://api.figma.com/v1/files/{file_key}/nodes?ids={ids_param}",
        headers={"X-Figma-Token": api_key}
    )

def test_get_nodes_info_api_error(monkeypatch, mock_requests_get):
    """Test get_figma_nodes_info with a Figma API error response."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key_error"
    node_ids = ["1:1"]
    ids_param = ",".join(node_ids)

    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 404 # Not Found
    mock_response.text = "Nodes not found"
    mock_requests_get.return_value = mock_response

    with pytest.raises(EffectiveHTTPException) as exc_info:
        get_figma_nodes_info(file_key, node_ids)

    assert exc_info.value.status_code == 404
    assert "Nodes not found" in exc_info.value.detail
    mock_requests_get.assert_called_once()

def test_get_nodes_info_request_exception(monkeypatch, mock_requests_get):
    """Test get_figma_nodes_info with a requests library exception."""
    api_key = "fake_api_key"
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', api_key)

    file_key = "test_file_key_req_exception"
    node_ids = ["1:1"]
    mock_requests_get.side_effect = requests.exceptions.Timeout("Request timed out")

    with pytest.raises(EffectiveHTTPException) as exc_info:
        get_figma_nodes_info(file_key, node_ids)

    assert exc_info.value.status_code == 503 # Service Unavailable
    assert "Request timed out" in exc_info.value.detail
    mock_requests_get.assert_called_once()

def test_get_nodes_info_empty_node_ids_list(monkeypatch, mock_requests_get):
    """Test get_figma_nodes_info with an empty list of node_ids."""
    # API key can be set or not, behavior should be the same: no API call for empty node_ids.
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', "fake_api_key_for_empty_test")

    file_key = "test_file_key_empty_nodes"
    result = get_figma_nodes_info(file_key, []) # Empty list of node_ids

    assert result == {"nodes": {}} # Expecting an empty nodes dictionary
    mock_requests_get.assert_not_called() # Crucially, no API call should be made.

def test_get_nodes_info_api_key_missing_creates_placeholders_for_unseen_nodes(monkeypatch, mock_requests_get):
    """Test dummy data for nodes includes placeholders for specific requested IDs not in DUMMY_NODES_INFO_RESPONSE."""
    monkeypatch.setattr('backend.figma_retriever.FIGMA_API_KEY', None)

    file_key = "test_file_key"
    # "1:1" is in DUMMY_NODES_INFO_RESPONSE, "3:3" is not.
    node_ids = ["1:1", "3:3"]

    result = get_figma_nodes_info(file_key, node_ids)

    assert "nodes" in result
    assert "1:1" in result["nodes"] # This should come from DUMMY_NODES_INFO_RESPONSE
    assert result["nodes"]["1:1"]["document"]["name"] == "Frame1"

    # Current logic in figma_retriever.py: if some nodes are found in DUMMY_NODES_INFO_RESPONSE,
    # it does not create placeholders for other requested node_ids that are not found.
    # Placeholders are only created if *no* requested nodes are found in the main dummy data.
    # Therefore, "3:3" should NOT be in result["nodes"] because "1:1" was found.
    assert "3:3" not in result["nodes"], \
        "Placeholder for '3:3' should not be created when other nodes like '1:1' are found in dummy data."

    mock_requests_get.assert_not_called()

# Note: The FIGMA_API_KEY is patched at the module level in figma_retriever.py
# This is because figma_retriever.py tries to load it from backend.config or os.getenv at import time.
# Monkeypatching `backend.figma_retriever.FIGMA_API_KEY` directly targets this loaded global.
# This is generally effective for module-level constants that are set once at import.
# If figma_retriever.py were to re-evaluate FIGMA_API_KEY from config/env inside each function,
# then monkeypatching backend.config.FIGMA_API_KEY or os.environ would be the way.
# Given current figma_retriever.py structure, this direct patch should work.
# The `figma_retriever.py` also has a print statement for when backend.config cannot be imported.
# This is fine for testing and indicates the fallback path is taken if tests are run in certain ways.
# The `EffectiveHTTPException` handles cases where FastAPI might not be fully available in a minimal test env.
