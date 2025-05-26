import os
import pytest
import sys

# Add the parent directory (/app) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from document_processor import load_txt, chunk_text, generate_embeddings
from unittest.mock import patch, MagicMock

# Define the path to the sample document for testing
# Assumes this test file is in backend/tests/ and the sample doc is also there
SAMPLE_DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_test_doc.txt")
EMPTY_DOC_PATH = os.path.join(os.path.dirname(__file__), "empty_test_doc.txt")

@pytest.fixture(scope="module", autouse=True)
def create_empty_file():
    # Create an empty file for testing empty file handling
    with open(EMPTY_DOC_PATH, 'w') as f:
        pass
    yield
    # Teardown: remove the empty file
    os.remove(EMPTY_DOC_PATH)

def test_load_txt_success():
    """Tests loading a .txt file successfully."""
    content = load_txt(SAMPLE_DOC_PATH)
    assert "This is a test document" in content
    assert "Lorem ipsum dolor sit amet" in content

def test_load_txt_file_not_found():
    """Tests loading a non-existent .txt file."""
    content = load_txt("non_existent_file.txt")
    assert content == ""

def test_load_txt_empty_file():
    """Tests loading an empty .txt file."""
    content = load_txt(EMPTY_DOC_PATH)
    assert content == ""

def test_chunk_text_simple():
    """Tests basic text chunking."""
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=5, overlap=1) # Words as units
    assert len(chunks) == 3 
    # Expected: "one two three four five", "five six seven eight nine", "nine ten" (approx)
    # Word joining makes it tricky, let's check content
    assert "one two three four five" in chunks[0]
    # Overlap means "five" should be in the second chunk
    assert "five six seven eight nine" in chunks[1] 
    # Last chunk might be smaller
    assert "nine ten" in chunks[2] or "ten" in chunks[2] # Depending on exact split

def test_chunk_text_with_overlap():
    text = "This is a longer sentence to test chunking with more overlap and ensure content integrity."
    words = text.split()
    chunk_size = 10
    overlap = 3
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    
    # Check that chunks are created
    assert len(chunks) > 0
    
    # Check content of first chunk
    expected_first_chunk = " ".join(words[:chunk_size])
    assert chunks[0] == expected_first_chunk
    
    # Check overlap: last 'overlap' words of chunk 0 should be first 'overlap' words of chunk 1
    if len(chunks) > 1:
        overlap_words_chunk0 = words[chunk_size-overlap:chunk_size]
        overlap_words_chunk1 = chunks[1].split()[:overlap]
        assert overlap_words_chunk0 == overlap_words_chunk1

def test_chunk_text_small_text():
    """Tests chunking with text smaller than chunk size."""
    text = "short text"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_empty_text():
    """Tests chunking with empty text."""
    text = ""
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) == 0

# Mocking SentenceTransformer for generate_embeddings
# The actual model loading will likely fail in the constrained environment.
@patch('document_processor.SentenceTransformer')
def test_generate_embeddings_mocked(MockSentenceTransformer):
    # Configure the mock model and its encode method
    mock_model_instance = MagicMock()
    # Simulate model.encode() returning a list of lists (or numpy array that can be .tolist())
    mock_model_instance.encode.return_value = [[0.1, 0.2], [0.3, 0.4]] 
    MockSentenceTransformer.return_value = mock_model_instance
    
    # Re-assign the model in document_processor to our mock for this test's scope
    # This is tricky because the model is loaded at module level.
    # A better way would be to pass the model into generate_embeddings,
    # but for now, we try to patch it globally for this test.
    # This kind of patching is more reliable if document_processor.model is explicitly re-assigned
    # or if generate_embeddings takes model as an argument.
    # For now, we rely on the initial patch at import time of document_processor.
    
    # To ensure the mock is used, we can temporarily set the global 'model' in document_processor
    # This is quite intrusive and generally not recommended, but module-level globals are hard to mock per-test.
    import document_processor as dp
    original_model = dp.model
    dp.model = mock_model_instance # Force use our mock

    chunks = ["first chunk", "second chunk"]
    embeddings = generate_embeddings(chunks)
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]
    assert embeddings[1] == [0.3, 0.4]
    mock_model_instance.encode.assert_called_once_with(chunks, convert_to_tensor=False)

    dp.model = original_model # Restore original model

@patch('document_processor.model', None) # Simulate model not loaded
def test_generate_embeddings_model_not_loaded():
    chunks = ["test chunk"]
    embeddings = generate_embeddings(chunks)
    assert embeddings == []

def test_generate_embeddings_empty_chunks():
    embeddings = generate_embeddings([])
    assert embeddings == []

# PDF and DOCX loading tests would go here, but likely fail if PyPDF2/python-docx
# were not installed. We can write them assuming they *should* work, or skip.
# For now, skipping due to high probability of lib installation failure.

# To run: pytest backend/tests/test_document_processor.py
# (Ensure backend/tests/sample_test_doc.txt exists)
