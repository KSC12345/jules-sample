import pytest
import os
import sys
from unittest.mock import patch, MagicMock, call

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Module to be tested
import backend.vector_store as vs

# Mock the global chromadb client and collection in vector_store.py
# This allows us to control their behavior during tests without a live DB.
mock_chroma_client = MagicMock()
mock_chroma_collection = MagicMock()

# Apply these mocks to the vector_store module before tests run
# This is a common way to handle module-level globals in testing.
vs.client = mock_chroma_client
vs.collection = mock_chroma_collection
vs.chromadb = MagicMock() # Mock the chromadb import itself if client init is complex

# --- Fixtures ---
@pytest.fixture(autouse=True)
def reset_mocks_before_each_test():
    """Ensures mocks are reset before each test function."""
    mock_chroma_client.reset_mock()
    mock_chroma_collection.reset_mock()
    
    # Set default return values or behaviors for the collection
    mock_chroma_collection.count.return_value = 0
    mock_chroma_collection.add.return_value = True # Assume add is successful by default
    mock_chroma_collection.query.return_value = { # Default query response
        "ids": [["id1"]], 
        "documents": [["doc1"]], 
        "metadatas": [[{"source": "test"}]],
        "distances": [[0.1]]
    }
    # Ensure the collection is "not None" for checks in the tested code
    vs.collection = mock_chroma_collection


# --- Tests for add_embeddings_to_collection ---
def test_add_embeddings_to_collection_success():
    chunk_ids = ["id1", "id2"]
    text_chunks = ["text1", "text2"]
    embeddings = [[0.1], [0.2]]
    metadatas = [{"src": "s1"}, {"src": "s2"}]
    
    result = vs.add_embeddings_to_collection(chunk_ids, text_chunks, embeddings, metadatas)
    
    assert result is True
    mock_chroma_collection.add.assert_called_once_with(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=text_chunks,
        metadatas=metadatas
    )

def test_add_embeddings_to_collection_mismatch_lengths():
    result = vs.add_embeddings_to_collection(["id1"], ["text1", "text2"], [[0.1]], [{"s":1}])
    assert result is False
    mock_chroma_collection.add.assert_not_called()

@patch('backend.vector_store.collection', None) # Simulate collection not initialized
def test_add_embeddings_to_collection_no_collection():
    result = vs.add_embeddings_to_collection(["id1"], ["text1"], [[0.1]], [{"s":1}])
    assert result is False

# --- Tests for query_collection_with_embedding ---
def test_query_collection_with_embedding_success():
    query_embedding = [0.1, 0.2, 0.3]
    n_results = 2
    
    expected_results = {"documents": [["doc1", "doc2"]]} # Simplified, matches default mock
    mock_chroma_collection.query.return_value = expected_results
    
    results = vs.query_collection_with_embedding(query_embedding, n_results=n_results)
    
    assert results == expected_results
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

@patch('backend.vector_store.collection', None)
def test_query_collection_with_embedding_no_collection():
    results = vs.query_collection_with_embedding([0.1], n_results=1)
    assert results is None

def test_query_collection_with_embedding_empty_embedding():
    results = vs.query_collection_with_embedding([], n_results=1)
    assert results is None
    mock_chroma_collection.query.assert_not_called()


# --- Tests for process_and_embed_documents ---
# These are more like integration tests for this module.
# We need to mock document_processor functions.
@patch('backend.vector_store.load_txt')
@patch('backend.vector_store.load_pdf') # Mock even if not directly testing PDF
@patch('backend.vector_store.load_docx') # Mock even if not directly testing DOCX
@patch('backend.vector_store.chunk_text')
@patch('backend.vector_store.generate_embeddings')
@patch('backend.vector_store.add_embeddings_to_collection') # Mock the function within vector_store itself
@patch('os.listdir')
@patch('os.path.join', side_effect=lambda *args: "/".join(args)) # Simple mock for os.path.join
@patch('os.path.isdir')
@patch('os.path.splitext')
def test_process_and_embed_documents_txt_file(
    mock_splitext, mock_isdir, mock_path_join, mock_listdir,
    mock_add_embed_coll, mock_gen_embed, mock_chunk_text,
    mock_load_docx, mock_load_pdf, mock_load_txt
):
    # Setup mocks
    mock_listdir.return_value = ["test_doc.txt"]
    mock_isdir.return_value = True # Assume directory exists
    mock_splitext.return_value = ("test_doc", ".txt") # For os.path.splitext(filename)[1].lower()
    
    mock_load_txt.return_value = "This is text content."
    mock_chunk_text.return_value = ["chunk1", "chunk2"]
    mock_gen_embed.return_value = [[0.1], [0.2]] # Embeddings for 2 chunks
    mock_add_embed_coll.return_value = True # Assume adding to collection is successful
    mock_chroma_collection.count.side_effect = [0, 2] # Initial count, count after adding 2 chunks

    docs_dir = "dummy_dir/docs"
    result = vs.process_and_embed_documents(documents_dir=docs_dir)

    assert result["status"] == "success"
    assert result["processed_files"] == 1
    assert result["total_chunks_added"] == 2
    assert result["collection_total_items"] == 2 # Based on side_effect for count

    mock_listdir.assert_called_once_with(docs_dir if os.path.isabs(docs_dir) else "/app/" + docs_dir)
    mock_load_txt.assert_called_once_with(f"{docs_dir if os.path.isabs(docs_dir) else '/app/'+docs_dir}/test_doc.txt")
    mock_chunk_text.assert_called_once_with("This is text content.", chunk_size=100, overlap=10)
    mock_gen_embed.assert_called_once_with(["chunk1", "chunk2"])
    
    expected_chunk_ids = ["test_doc.txt_chunk_0", "test_doc.txt_chunk_1"]
    expected_metadatas = [
        {"source_file": "test_doc.txt", "chunk_number": 0, "original_extension": ".txt"},
        {"source_file": "test_doc.txt", "chunk_number": 1, "original_extension": ".txt"}
    ]
    mock_add_embed_coll.assert_called_once_with(
        expected_chunk_ids,
        ["chunk1", "chunk2"],
        [[0.1], [0.2]],
        expected_metadatas
    )

@patch('backend.vector_store.collection', None) # Simulate collection not initialized
def test_process_and_embed_documents_no_collection():
    result = vs.process_and_embed_documents()
    assert result["status"] == "error"
    assert "ChromaDB collection or SentenceTransformer model is not initialized" in result["message"]

@patch('os.path.isdir', return_value=False) # Directory does not exist
def test_process_and_embed_documents_dir_not_found(mock_isdir):
    result = vs.process_and_embed_documents(documents_dir="non_existent_path")
    assert result["status"] == "error"
    assert "Directory 'non_existent_path' not found" in result["message"] or \
           "Directory '/app/non_existent_path' not found" in result["message"]


# To run: pytest backend/tests/test_vector_store.py
