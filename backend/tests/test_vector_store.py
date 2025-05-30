import pytest
import os
import shutil
import tempfile
import time

# Ensure backend modules are discoverable by Python.
# This is typically handled by running pytest with `python -m pytest` from the /app directory
# or by having PYTHONPATH configured correctly.
# If direct `python backend/tests/test_vector_store.py` is run, sys.path manipulation might be needed here:
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Fixtures ---

@pytest.fixture(scope="function")
def temp_chroma_dir(tmp_path_factory):
    """Creates a temporary directory for ChromaDB data for the duration of a test function."""
    path = tmp_path_factory.mktemp(f"chroma_test_data_{int(time.time() * 1000)}")
    print(f"Created temp_chroma_dir: {path}")
    yield str(path)
    # tmp_path_factory handles cleanup of the directory itself.
    # We might need to explicitly ensure chromadb releases any file locks if issues arise,
    # but usually, shutil.rmtree (done by tmp_path_factory) is sufficient after processes close.
    print(f"Cleaned up temp_chroma_dir: {path}")


@pytest.fixture(scope="function")
def vector_store_module(monkeypatch, temp_chroma_dir):
    """
    Provides an instance of the vector_store module with its configuration patched
    to use a temporary ChromaDB path and a unique, temporary collection name.
    This ensures test isolation.
    """
    # Generate a unique collection name for each test function to ensure isolation
    # even if the underlying client/DB path were somehow shared (it shouldn't be with temp_chroma_dir).
    test_collection_name = f"test_collection_{os.getpid()}_{int(time.time() * 1000)}"
    
    print(f"Patching config: CHROMA_DB_PATH='{temp_chroma_dir}', CHROMA_COLLECTION_NAME='{test_collection_name}'")
    
    # Patch backend.config attributes that vector_store.py uses at its import time.
    monkeypatch.setattr('backend.config.CHROMA_DB_PATH', temp_chroma_dir)
    monkeypatch.setattr('backend.config.CHROMA_COLLECTION_NAME', test_collection_name)
    
    # IMPORTANT: Because vector_store.py initializes its client and collection as global
    # variables at the module's top level (when it's first imported), simply patching
    # backend.config won't affect the already initialized client/collection if vector_store.py
    # was imported before this fixture ran (e.g., by another test module or conftest.py).
    #
    # To handle this, we need to ensure vector_store.py is re-evaluated *after*
    # the monkeypatching is in place for the scope of this test.
    # We can achieve this by using importlib.reload.
    import importlib
    import backend.vector_store # Ensure it's loaded once if not already
    
    # Store original values to restore later if necessary, though monkeypatch handles this for config.
    # However, for vector_store's own globals, we might need manual reset if not reloading.
    original_client = getattr(backend.vector_store, 'client', None)
    original_collection = getattr(backend.vector_store, 'collection', None)
    original_embedding_model = getattr(backend.vector_store, 'embedding_model', None)

    # Reload the module to re-execute its top-level assignments with patched config.
    reloaded_vs_module = importlib.reload(backend.vector_store)
    
    # Check if the reloaded module's client actually used the patched path.
    # This is a bit of an internal check, ideally, we verify behavior, not implementation.
    # However, it's crucial for test isolation.
    # Note: reloaded_vs_module.client.settings.chroma_db_impl etc. for new Chroma versions.
    # For older chromadb.PersistentClient(path=...), the path is stored.
    # Let's assume the print statements during initialization in vector_store.py will show the path.
    
    yield reloaded_vs_module # Provide the reloaded module to the tests

    # Teardown: Clean up the specific collection from the temporary ChromaDB instance.
    # The directory `temp_chroma_dir` itself is managed by tmp_path_factory.
    print(f"Attempting cleanup of collection '{test_collection_name}' from path '{temp_chroma_dir}'")
    try:
        if reloaded_vs_module.client:
            # Check if collection exists before trying to delete
            # This check might vary based on ChromaDB version API
            # For now, simple try-except for delete.
            try:
                reloaded_vs_module.client.delete_collection(name=test_collection_name)
                print(f"Successfully deleted test collection: {test_collection_name}")
            except Exception as e:
                # It might fail if collection was never created or already cleaned.
                print(f"Info/Warning during collection deletion: {e}")
        
        # Attempt to reset or close the client if possible, to release file locks before dir removal.
        # This is highly dependent on the ChromaDB client library's API.
        # if hasattr(reloaded_vs_module.client, 'reset'): reloaded_vs_module.client.reset()
        # if hasattr(reloaded_vs_module.client, 'close'): reloaded_vs_module.client.close()

    except Exception as e:
        print(f"Error during vector_store_module fixture teardown: {e}")
    
    # Restore original module state if it was significantly altered beyond monkeypatch.
    # This is more for modules that don't get reloaded but just patched.
    # Since we reloaded, other modules importing vector_store will get this reloaded version too
    # for the scope of the test session unless we manage sys.modules carefully.
    # For function-scoped fixture and reload, this should be okay.
    if original_client is not None: backend.vector_store.client = original_client
    if original_collection is not None: backend.vector_store.collection = original_collection
    if original_embedding_model is not None: backend.vector_store.embedding_model = original_embedding_model
    # We might need to reload again to restore original state if other tests depend on original config.
    # This highlights complexity of testing modules with global state initialized on import.


@pytest.fixture
def sample_documents_for_vs():
    """Provides a list of sample documents structured for vector store functions."""
    return [
        {"content": "React Button: A clickable UI element.", "metadata": {"filename": "Button.jsx", "id": "btn1"}},
        {"content": "React Card: Displays content in a container.", "metadata": {"filename": "Card.jsx", "id": "card1"}},
        {"content": "JavaScript utility for date manipulation.", "metadata": {"filename": "dateUtils.js", "id": "util1"}},
    ]

# --- Test Cases ---

def test_module_import_and_initialization(vector_store_module):
    """Checks if the vector_store module and its components are initialized."""
    vs = vector_store_module
    assert vs.embedding_model is not None, "Embedding model should be initialized."
    assert vs.client is not None, "ChromaDB client should be initialized."
    assert vs.collection is not None, "ChromaDB collection should be initialized."
    # The collection name should match what was set by monkeypatch
    assert vs.collection.name == vs.config.CHROMA_COLLECTION_NAME 
    print(f"Test collection name in use: {vs.collection.name}")
    # Path check is harder here, would rely on printed output from vector_store.py for confirmation
    # or direct inspection of vs.client if its API allows.


def test_add_documents(vector_store_module, sample_documents_for_vs):
    """Test adding documents to an isolated ChromaDB instance."""
    vs = vector_store_module
    
    initial_count = vs.collection.count()
    assert initial_count == 0, "Collection should be empty at the start of this test."
    
    success = vs.add_documents(sample_documents_for_vs)
    assert success, "add_documents should return True."
    
    assert vs.collection.count() == len(sample_documents_for_vs), \
        f"Collection count should be {len(sample_documents_for_vs)} after adding."

    # Verify one document by getting it (IDs are f"{collection_name}_{filename}")
    test_doc = sample_documents_for_vs[0]
    doc_id_in_db = f"{vs.config.CHROMA_COLLECTION_NAME}_{test_doc['metadata']['filename']}"
    retrieved = vs.collection.get(ids=[doc_id_in_db], include=["documents", "metadatas"])
    
    assert len(retrieved['ids']) == 1, "Should retrieve the added document by ID."
    assert retrieved['ids'][0] == doc_id_in_db
    assert retrieved['documents'][0] == test_doc['content']
    # ChromaDB stringifies all metadata values.
    assert retrieved['metadatas'][0]['filename'] == str(test_doc['metadata']['filename'])
    assert retrieved['metadatas'][0]['id'] == str(test_doc['metadata']['id'])


def test_query_documents(vector_store_module, sample_documents_for_vs):
    """Test querying documents after adding them."""
    vs = vector_store_module
    vs.add_documents(sample_documents_for_vs)
    assert vs.collection.count() == len(sample_documents_for_vs), "Documents not added correctly."

    query_text = "Clickable button element"
    n_results = 1
    results = vs.query_documents(query_text, n_results=n_results)
    
    assert len(results) == n_results, f"Should return {n_results} result(s)."
    for doc in results:
        assert "content" in doc and "metadata" in doc and "distance" in doc
    
    # Expect Button.jsx to be the top result for this query
    assert results[0]['metadata']['filename'] == "Button.jsx", \
        f"Expected Button.jsx for query '{query_text}', got {results[0]['metadata']['filename']}"
    print(f"Query: '{query_text}', Top result: {results[0]['metadata']['filename']}, Distance: {results[0]['distance']:.4f}")


def test_query_empty_store(vector_store_module):
    """Test querying an empty store."""
    vs = vector_store_module
    assert vs.collection.count() == 0, "Collection must be empty for this test."
    
    results = vs.query_documents("query for empty store", n_results=3)
    assert len(results) == 0, "Querying an empty store should yield no results."


def test_add_documents_empty_list(vector_store_module):
    """Test adding an empty list of documents."""
    vs = vector_store_module
    success = vs.add_documents([])
    assert success, "Adding an empty list should be a no-op and return True."
    assert vs.collection.count() == 0


def test_add_documents_with_invalid_structure(vector_store_module):
    """Test adding documents with missing 'content' or 'metadata'."""
    vs = vector_store_module
    invalid_docs = [
        {"metadata": {"filename": "missing_content.jsx"}}, # Missing content
        {"content": "no metadata here"}, # Missing metadata
        {"content": "no filename in metadata", "metadata": {"type": "component"}} # Missing filename
    ]
    # The add_documents function currently prints warnings and skips invalid docs.
    # It should still return True if some docs were processed, or False if all failed AND it's designed to.
    # Current implementation: filters out bad ones, proceeds with good ones. If all bad, returns False.
    # Let's test with only bad ones.
    success = vs.add_documents(invalid_docs)
    assert not success, "Should return False if all documents are invalid and none are added."
    assert vs.collection.count() == 0, "No documents should be added if all are invalid."

    # Test with a mix
    valid_doc = {"content": "Valid content", "metadata": {"filename": "valid.jsx"}}
    mixed_docs = invalid_docs + [valid_doc]
    success_mixed = vs.add_documents(mixed_docs)
    assert success_mixed, "Should return True if at least one valid document is processed."
    assert vs.collection.count() == 1, "Only one valid document should be added."
    retrieved = vs.collection.get(include=["documents"])
    assert retrieved['documents'][0] == "Valid content"


# To run tests from /app directory:
# python -m pytest backend/tests
# (Ensure relevant models for SentenceTransformer are downloadable/cached if not mocked,
# though vector_store.py initializes its own SentenceTransformer instance.)
# The SentenceTransformer model is loaded when vector_store.py is imported/reloaded.
# This might cause downloads on first run or in CI.
# For true unit tests of vector_store logic, SentenceTransformer could also be mocked,
# but testing with the real model for ChromaDB interaction is more of an integration test, which is valuable here.
