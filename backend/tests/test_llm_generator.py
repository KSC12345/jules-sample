import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.llm_generator import generate_response_from_context

# Mock the global pipeline and tokenizer variables in llm_generator
# Patch 'backend.llm_generator.generator_pipeline' and 'backend.llm_generator.tokenizer'
@patch('backend.llm_generator.generator_pipeline')
@patch('backend.llm_generator.tokenizer')
def test_generate_response_with_context(MockTokenizer, MockPipeline):
    # Configure the mock pipeline
    # Simulate the pipeline returning a dictionary (or list of dicts)
    # where 'generated_text' is a key.
    mock_generated_output = [{"generated_text": "Context: Test context.\nQuestion: Test query\nAnswer: This is a mock answer."}]
    MockPipeline.return_value = mock_generated_output
    
    # Configure the mock tokenizer (needed for len(tokenizer.encode(prompt)))
    # and if tokenizer.pad_token is None checks, etc.
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.eos_token = "<|endoftext|>" # Example token
    mock_tokenizer_instance.pad_token = mock_tokenizer_instance.eos_token
    mock_tokenizer_instance.encode.return_value = [1, 2, 3] # Dummy encoded prompt
    MockTokenizer = mock_tokenizer_instance # Assign configured mock

    # Re-assign the global variables in llm_generator to our mocks for this test
    # This is essential because the function generate_response_from_context uses these globals.
    import backend.llm_generator as lg
    original_pipeline = lg.generator_pipeline
    original_tokenizer = lg.tokenizer
    lg.generator_pipeline = MockPipeline
    lg.tokenizer = MockTokenizer

    query = "Test query"
    context_chunks = ["Test context chunk 1.", "Another piece of context."]
    
    response = generate_response_from_context(query, context_chunks)

    assert response == "This is a mock answer."
    
    # Check that the pipeline was called
    # The exact prompt construction needs to be verified if it's complex.
    # For now, just check it was called.
    MockPipeline.assert_called_once()
    call_args = MockPipeline.call_args
    prompt_arg = call_args[0][0] # First positional argument of the call
    
    assert "Context:\nTest context chunk 1.\nAnother piece of context." in prompt_arg
    assert f"Question: {query}" in prompt_arg

    # Restore original globals
    lg.generator_pipeline = original_pipeline
    lg.tokenizer = original_tokenizer


@patch('backend.llm_generator.generator_pipeline')
@patch('backend.llm_generator.tokenizer')
def test_generate_response_no_context(MockTokenizer, MockPipeline):
    mock_generated_output = [{"generated_text": "Question: Test query no context\nAnswer: Mock answer for no context."}]
    MockPipeline.return_value = mock_generated_output
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.eos_token = "<|endoftext|>"
    mock_tokenizer_instance.pad_token = mock_tokenizer_instance.eos_token
    mock_tokenizer_instance.encode.return_value = [1,2,3]
    MockTokenizer = mock_tokenizer_instance

    import backend.llm_generator as lg
    original_pipeline = lg.generator_pipeline
    original_tokenizer = lg.tokenizer
    lg.generator_pipeline = MockPipeline
    lg.tokenizer = MockTokenizer

    query = "Test query no context"
    response = generate_response_from_context(query, [])

    assert response == "Mock answer for no context."
    MockPipeline.assert_called_once()
    call_args = MockPipeline.call_args
    prompt_arg = call_args[0][0]
    assert "Context:" not in prompt_arg # No context should be in the prompt
    assert f"Question: {query}" in prompt_arg

    lg.generator_pipeline = original_pipeline
    lg.tokenizer = original_tokenizer


@patch('backend.llm_generator.generator_pipeline', None) # Simulate pipeline not loaded
def test_generate_response_pipeline_not_available():
    query = "Test query"
    context_chunks = ["Test context."]
    response = generate_response_from_context(query, context_chunks)
    assert "Error: Text generation pipeline is not available" in response


@patch('backend.llm_generator.generator_pipeline')
@patch('backend.llm_generator.tokenizer')
def test_generate_response_long_context_truncation(MockTokenizer, MockPipeline):
    mock_generated_output = [{"generated_text": "Context: Shortened context...\nQuestion: Test query\nAnswer: Mock response for truncated."}]
    MockPipeline.return_value = mock_generated_output

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.eos_token = "<|endoftext|>"
    mock_tokenizer_instance.pad_token = mock_tokenizer_instance.eos_token
    mock_tokenizer_instance.encode.return_value = [1,2,3]
    MockTokenizer = mock_tokenizer_instance
    
    import backend.llm_generator as lg
    original_pipeline = lg.generator_pipeline
    original_tokenizer = lg.tokenizer
    lg.generator_pipeline = MockPipeline
    lg.tokenizer = MockTokenizer

    query = "Test query with long context"
    # Create a very long context chunk
    long_chunk = "This is a very long string. " * 200 # Approx 20 chars * 200 = 4000 chars
    context_chunks = [long_chunk]
    max_context_len = 1500 # As defined in llm_generator

    response = generate_response_from_context(query, context_chunks, max_context_length=max_context_len)
    
    assert response == "Mock response for truncated."
    MockPipeline.assert_called_once()
    call_args = MockPipeline.call_args
    prompt_arg = call_args[0][0] # First positional argument

    # Check that the context in the prompt is truncated
    # "Context:\n" is 9 chars. "Question: ..." part is also there.
    # The actual context part should be around max_context_len.
    # A bit hard to assert exact length due to "Context:\n" and other parts of prompt.
    # But we can check if the original long_chunk (which is > max_context_len) is NOT fully there.
    assert len(prompt_arg.split("Context:\n")[1].split("\n\nQuestion:")[0]) <= max_context_len
    assert long_chunk not in prompt_arg # The full long chunk should not be present

    lg.generator_pipeline = original_pipeline
    lg.tokenizer = original_tokenizer

# To run: pytest backend/tests/test_llm_generator.py
