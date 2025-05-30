import requests
import os # For potential future use, not strictly needed for FIGMA_API_KEY via config
import json # For parsing JSON responses and crafting dummy data
from typing import List, Dict, Any, Optional

# Attempt to import FIGMA_API_KEY from backend.config
# Also import related configuration for consistency if needed later
try:
    from . import config as backend_config
    FIGMA_API_KEY = backend_config.FIGMA_API_KEY
    # OPENAI_API_KEY = backend_config.OPENAI_API_KEY # Example if other keys were needed
except ImportError:
    # This fallback is for direct execution of this script, e.g., during development/testing,
    # where '.config' might not be resolvable if 'backend' is not in PYTHONPATH or CWD is backend/.
    # It tries to load FIGMA_API_KEY directly from environment variables as a last resort.
    print("Could not import backend.config, attempting to load FIGMA_API_KEY directly from environment.")
    from dotenv import load_dotenv
    load_dotenv() # Load .env file if present (especially for local testing)
    FIGMA_API_KEY = os.getenv("FIGMA_API_KEY")
    if FIGMA_API_KEY is None:
        print("Warning: FIGMA_API_KEY not found in environment variables after direct load attempt.")


# Base URL for the Figma API
FIGMA_API_BASE_URL = "https://api.figma.com/v1/"

# --- Dummy Data ---
DUMMY_FILE_INFO_RESPONSE = {
  "name": "Dummy Figma File",
  "role": "editor",
  "lastModified": "2024-05-30T10:00:00Z",
  "thumbnailUrl": "",
  "version": "123",
  "document": {
    "id": "0:0",
    "name": "Root Document Node",
    "type": "DOCUMENT",
    "children": [
      {
        "id": "1:1",
        "name": "Frame1",
        "type": "FRAME",
        "blendMode": "PASS_THROUGH",
        "children": [
            {
                "id": "1:2",
                "name": "Rectangle1",
                "type": "RECTANGLE",
                "blendMode": "PASS_THROUGH",
                "absoluteBoundingBox": {"x": 10, "y": 10, "width": 50, "height": 50},
                "fills": [{"type": "SOLID", "color": {"r": 0.8, "g": 0.8, "b": 0.8, "a": 1}}],
                "strokes": [],
                "strokeWeight": 1,
                "opacity": 1,
                "effects": []
            }
        ],
        "absoluteBoundingBox": {"x": 0, "y": 0, "width": 100, "height": 100},
        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}],
        "strokes": [],
        "strokeWeight": 1,
        "opacity": 1,
        "effects": []
      },
      {
        "id": "2:1",
        "name": "Frame2",
        "type": "FRAME",
        "children": [],
        "absoluteBoundingBox": {"x": 150, "y": 0, "width": 200, "height": 150},
        "fills": [{"type": "SOLID", "color": {"r": 0, "g": 1, "b": 0, "a": 1}}],
        "strokes": [],
        "strokeWeight": 1,
        "opacity": 1,
        "effects": []
      }
    ]
  },
  "components": {},
  "componentSets": {},
  "schemaVersion": 0,
  "styles": {}
}

DUMMY_NODES_INFO_RESPONSE = {
  "name": "Dummy Figma File (Node Info)",
  "role": "viewer",
  "lastModified": "2024-05-30T10:00:00Z",
  "thumbnailUrl": "",
  "version": "123",
  "err": None,
  "nodes": {
    "1:1": {
      "document": {
        "id": "1:1",
        "name": "Frame1",
        "type": "FRAME",
        "blendMode": "PASS_THROUGH",
        "children": [ # Children might or might not be included depending on API call details
             {
                "id": "1:2",
                "name": "Rectangle1",
                "type": "RECTANGLE",
             }
        ],
        "absoluteBoundingBox": {"x": 0, "y": 0, "width": 100, "height": 100},
        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 0, "b": 0, "a": 1}}],
        "strokes": [],
        "strokeWeight": 1,
        "opacity": 1,
        "effects": []
      },
      "components": {},
      "componentSets": {},
      "schemaVersion": 0,
      "styles": {}
    },
    "1:2": { # Example if a second node ID was requested
      "document": {
        "id": "1:2",
        "name": "Rectangle1",
        "type": "RECTANGLE",
        "blendMode": "PASS_THROUGH",
        "children": [],
        "absoluteBoundingBox": {"x": 10, "y": 10, "width": 50, "height": 50},
        "fills": [{"type": "SOLID", "color": {"r": 0.8, "g": 0.8, "b": 0.8, "a": 1}}],
        "strokes": [],
        "strokeWeight": 1,
        "opacity": 1,
        "effects": []
      },
      "components": {},
      "componentSets": {},
      "schemaVersion": 0,
      "styles": {}
    }
  }
}

# Placeholder for HTTPException if not in FastAPI context (e.g. direct script run)
class PlaceholderHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")

# Try to import real HTTPException from FastAPI, fallback to placeholder
try:
    from fastapi import HTTPException
except ImportError:
    HTTPException = PlaceholderHTTPException
    print("Warning: HTTPException from FastAPI not found. Using placeholder for direct script execution.")


def get_figma_file_info(file_key: str) -> Dict[str, Any]:
    """
    Retrieves information about a Figma file.

    Args:
        file_key (str): The key of the Figma file.

    Returns:
        Dict[str, Any]: The parsed JSON response from the Figma API,
                        or a dummy response if API key is not configured.

    Raises:
        HTTPException: If the API call fails (e.g. non-200 status code)
                       and FIGMA_API_KEY is configured.
    """
    if not FIGMA_API_KEY:
        print(f"Warning: FIGMA_API_KEY not set. Returning dummy data for file_key: {file_key}")
        return DUMMY_FILE_INFO_RESPONSE

    url = f"{FIGMA_API_BASE_URL}files/{file_key}"
    headers = {"X-Figma-Token": FIGMA_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = f"Figma API Error for file_key {file_key}: {response.status_code} - {response.text}"
            print(error_detail)
            # In a FastAPI app, this would raise an exception caught by the framework.
            # When run directly, it will raise the PlaceholderHTTPException if FastAPI isn't installed.
            raise HTTPException(status_code=response.status_code, detail=error_detail)
    except requests.exceptions.RequestException as e:
        error_detail = f"Request failed for Figma file_key {file_key}: {e}"
        print(error_detail)
        raise HTTPException(status_code=503, detail=error_detail) # Service Unavailable


def get_figma_nodes_info(file_key: str, node_ids: List[str]) -> Dict[str, Any]:
    """
    Retrieves information about specific nodes within a Figma file.

    Args:
        file_key (str): The key of the Figma file.
        node_ids (List[str]): A list of node IDs to retrieve.

    Returns:
        Dict[str, Any]: The parsed JSON response from the Figma API,
                        or a dummy response if API key is not configured or node_ids is empty.

    Raises:
        HTTPException: If the API call fails (e.g. non-200 status code)
                       and FIGMA_API_KEY is configured.
    """
    if not node_ids:
        print("Warning: node_ids list is empty. Returning empty nodes data.")
        # Return a structure similar to Figma's but with empty nodes, or a specific message.
        return {"nodes": {}}

    if not FIGMA_API_KEY:
        print(f"Warning: FIGMA_API_KEY not set. Returning dummy data for nodes in file_key: {file_key}")
        # Filter dummy nodes to only include requested IDs if possible, or return all dummy nodes.
        # For simplicity, let's return a modified DUMMY_NODES_INFO_RESPONSE that only contains
        # nodes from the DUMMY_NODES_INFO_RESPONSE that match the requested node_ids (if any).
        # This makes the dummy data slightly more responsive to the input.

        filtered_dummy_nodes = {
            node_id: DUMMY_NODES_INFO_RESPONSE["nodes"][node_id]
            for node_id in node_ids if node_id in DUMMY_NODES_INFO_RESPONSE["nodes"]
        }
        if not filtered_dummy_nodes and node_ids: # If specific IDs were requested but not in main dummy set
             # Create placeholder nodes for requested IDs if not found in the main dummy set
            for node_id in node_ids:
                if node_id not in filtered_dummy_nodes:
                    filtered_dummy_nodes[node_id] = {
                        "document": {
                            "id": node_id, "name": f"Dummy Node {node_id}", "type": "RECTANGLE",
                            "absoluteBoundingBox": {"x":0,"y":0,"width":10,"height":10}, "fills":[]
                        },
                        "components": {}, "schemaVersion": 0, "styles": {}
                    }
        return {"nodes": filtered_dummy_nodes, "name": DUMMY_NODES_INFO_RESPONSE["name"]}


    ids_param = ",".join(node_ids)
    url = f"{FIGMA_API_BASE_URL}files/{file_key}/nodes?ids={ids_param}"
    headers = {"X-Figma-Token": FIGMA_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = f"Figma API Error for nodes in file_key {file_key} (ids: {ids_param}): {response.status_code} - {response.text}"
            print(error_detail)
            raise HTTPException(status_code=response.status_code, detail=error_detail)
    except requests.exceptions.RequestException as e:
        error_detail = f"Request failed for Figma nodes in file_key {file_key} (ids: {ids_param}): {e}"
        print(error_detail)
        raise HTTPException(status_code=503, detail=error_detail) # Service Unavailable

if __name__ == '__main__':
    print("--- Testing Figma Retriever Functions (expecting dummy data if API key is not set) ---")

    # Test get_figma_file_info
    print("\n--- Testing get_figma_file_info ---")
    try:
        file_info = get_figma_file_info(file_key="test_file_key_123")
        print("File Info (raw):")
        # Pretty print JSON for readability
        print(json.dumps(file_info, indent=2))
        if "document" in file_info and "children" in file_info["document"]:
             print(f"\nSuccessfully retrieved/mocked file info. Document has {len(file_info['document']['children'])} top-level children.")
    except PlaceholderHTTPException as e: # Catching the placeholder for direct run
        print(f"Caught PlaceholderHTTPException (expected if API call failed and FastAPI not installed): {e.status_code} - {e.detail}")
    except Exception as e: # Catch any other unexpected error
        print(f"An unexpected error occurred during get_figma_file_info: {e}")

    # Test get_figma_nodes_info
    print("\n--- Testing get_figma_nodes_info ---")
    dummy_node_ids = ["1:1", "1:2", "non_existent_node:99"] # include one that might not be in detailed dummy data
    try:
        nodes_info = get_figma_nodes_info(file_key="test_file_key_123", node_ids=dummy_node_ids)
        print("Nodes Info (raw):")
        print(json.dumps(nodes_info, indent=2))
        if "nodes" in nodes_info:
            print(f"\nSuccessfully retrieved/mocked nodes info. Found data for {len(nodes_info['nodes'])} out of {len(dummy_node_ids)} requested node(s).")
            for node_id, node_data in nodes_info["nodes"].items():
                print(f"  Node {node_id}: Name - {node_data.get('document', {}).get('name', 'N/A')}, Type - {node_data.get('document', {}).get('type', 'N/A')}")

    except PlaceholderHTTPException as e:
        print(f"Caught PlaceholderHTTPException: {e.status_code} - {e.detail}")
    except Exception as e:
        print(f"An unexpected error occurred during get_figma_nodes_info: {e}")

    print("\n--- Testing get_figma_nodes_info with empty node_ids list ---")
    try:
        empty_nodes_info = get_figma_nodes_info(file_key="test_file_key_123", node_ids=[])
        print("Empty Nodes Info (raw):")
        print(json.dumps(empty_nodes_info, indent=2))
        if "nodes" in empty_nodes_info and not empty_nodes_info["nodes"]:
            print("\nSuccessfully handled empty node_ids list.")
    except PlaceholderHTTPException as e:
        print(f"Caught PlaceholderHTTPException: {e.status_code} - {e.detail}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Example of how FIGMA_API_KEY is seen by the script
    if FIGMA_API_KEY:
        print(f"\nFigma API Key is configured (first 5 chars): {FIGMA_API_KEY[:5]}...")
    else:
        print("\nFigma API Key is NOT configured.")

    print("\n--- Figma Retriever Tests Complete ---")
