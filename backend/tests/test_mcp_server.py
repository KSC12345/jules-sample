import unittest
from unittest.mock import patch, MagicMock, ANY
import grpc # Required for grpc.StatusCode
import os # For os.path related mocks

# Adjust import path if necessary, assuming test is run from project root or backend/tests/
# If running with pytest from root, 'backend.mcp_server' should work.
# If running script directly from backend/tests/, sys.path manipulation might be needed in test.
try:
    from backend.mcp_server import MCPServiceServicer
    from backend import mcp_pb2
    # Mocked versions of shared resources for testing
    from backend.vector_store import collection as chromadb_collection_real
    from backend.llm_generator import generator_pipeline as llm_pipeline_real
    from backend.document_processor import generate_embeddings as generate_embeddings_real

except ImportError as e:
    print(f"Test setup: Error importing backend modules: {e}. Ensure PYTHONPATH is set correctly or run tests via pytest from root.")
    # Define placeholders if imports fail, so tests can be discovered but will likely fail with NameError
    MCPServiceServicer = None 
    mcp_pb2 = None


# Helper to create a mock gRPC context
def _create_mock_context():
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.set_code = MagicMock()
    mock_context.set_details = MagicMock()
    return mock_context

@unittest.skipIf(MCPServiceServicer is None or mcp_pb2 is None, "Skipping MCP server tests due to import errors during test discovery.")
class TestMCPServiceServicer(unittest.TestCase):

    def setUp(self):
        # Instantiate the servicer
        self.servicer = MCPServiceServicer()
        # Create a mock gRPC context for each test
        self.mock_context = _create_mock_context()

    @patch('backend.mcp_server.generate_query_embedding')
    @patch('backend.mcp_server.query_collection_with_embedding')
    @patch('backend.mcp_server.generate_response_from_context')
    @patch('backend.mcp_server.chromadb_collection', MagicMock()) # Mock as an object
    @patch('backend.mcp_server.llm_pipeline_global', MagicMock()) # Mock as an object
    def test_chat_successful(self, mock_generate_response, mock_query_db, mock_generate_embedding):
        # Setup mocks
        mock_generate_embedding.return_value = [[0.1, 0.2]]
        mock_query_db.return_value = {'documents': [['context chunk 1']]}
        mock_generate_response.return_value = "LLM response"

        # Create request
        request = mcp_pb2.MCPChatMessage(message="Hello")

        # Call RPC method
        response = self.servicer.Chat(request, self.mock_context)

        # Assertions
        self.assertEqual(response.user_query, "Hello")
        self.assertEqual(response.llm_response, "LLM response")
        self.assertIn("context chunk 1", response.retrieved_context_chunks)
        mock_generate_embedding.assert_called_once_with(["Hello"])
        mock_query_db.assert_called_once_with(query_embedding=[0.1, 0.2], n_results=3)
        mock_generate_response.assert_called_once_with(query="Hello", context_chunks=['context chunk 1'], max_context_length=1500, max_new_tokens=100)
        self.mock_context.set_code.assert_not_called()

    @patch('backend.mcp_server.chromadb_collection', None) # Simulate component not available
    def test_chat_missing_chromadb(self):
        request = mcp_pb2.MCPChatMessage(message="Hello")
        self.servicer.Chat(request, self.mock_context)
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.UNAVAILABLE)

    @patch('backend.mcp_server.generate_query_embedding', side_effect=Exception("Embedding error"))
    @patch('backend.mcp_server.chromadb_collection', MagicMock())
    @patch('backend.mcp_server.llm_pipeline_global', MagicMock())
    def test_chat_embedding_failure(self, mock_generate_embedding): # mock_generate_embedding must be passed if patched
        request = mcp_pb2.MCPChatMessage(message="Hello")
        # Corrected: The mock_generate_embedding is now a parameter
        response = self.servicer.Chat(request, self.mock_context)
        # Check if embedding generation was indeed called
        mock_generate_embedding.assert_called_once_with([request.message])
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        # The detail message was "Failed to generate embedding for the user message."
        # This needs to be updated if the actual error message is different or more generic
        # For now, let's assume the original detail message is what we expect.
        # However, if the exception is "Embedding error", the server might log "An unexpected error occurred: Embedding error"
        # Let's check the original code for mcp_server.py Chat method's exception handling for generate_query_embedding
        # Original server code:
        # if not query_embedding_list or not query_embedding_list[0]:
        #     context.set_code(grpc.StatusCode.INTERNAL)
        #     context.set_details("Failed to generate embedding for the user message.")
        # This is for when the *result* is bad. If it *throws an exception*, it falls into the general catch-all.
        # except Exception as e:
        #     print(f"MCP Server: Error during Chat RPC: {e}")
        #     context.set_code(grpc.StatusCode.INTERNAL)
        #     context.set_details(f"An unexpected error occurred: {str(e)}")
        # So, the detail should be "An unexpected error occurred: Embedding error"
        self.mock_context.set_details.assert_called_with("An unexpected error occurred: Embedding error")


    @patch('backend.mcp_server.process_docs_for_rag_internal')
    @patch('backend.mcp_server.os.path.isdir', return_value=True)
    @patch('backend.mcp_server.os.path.isabs') # Mock isabs to check its call
    @patch('backend.mcp_server.os.path.join')   # Mock join to check its call
    @patch('backend.mcp_server.os.getenv')     # Mock getenv
    @patch('backend.mcp_server.chromadb_collection', MagicMock())
    @patch('backend.mcp_server.generate_query_embedding', MagicMock()) 
    def test_process_documents_successful_relative_path_docker(self, mock_getenv, mock_join, mock_isabs, mock_isdir, mock_process_docs):
        # Simulate Docker environment and relative path
        mock_getenv.return_value = "true" # RUNNING_IN_DOCKER
        mock_isabs.return_value = False   # Path is relative
        mock_join.return_value = "/app/test/docs" # Expected joined path

        mock_process_docs.return_value = {
            "status": "success", "processed_files": 1, "total_chunks_added": 10, "collection_total_items": 100
        }
        request = mcp_pb2.MCPProcessRequest(directory="test/docs")
        response = self.servicer.ProcessDocuments(request, self.mock_context)

        mock_isabs.assert_called_once_with("test/docs")
        mock_getenv.assert_called_once_with("RUNNING_IN_DOCKER")
        mock_join.assert_called_once_with("/app", "test/docs")
        mock_isdir.assert_called_once_with("/app/test/docs")
        mock_process_docs.assert_called_once_with(documents_dir="/app/test/docs")
        
        self.assertEqual(response.status, "success")
        self.assertEqual(response.processed_files, 1)
        self.mock_context.set_code.assert_not_called()

    @patch('backend.mcp_server.process_docs_for_rag_internal')
    @patch('backend.mcp_server.os.path.isdir', return_value=True)
    @patch('backend.mcp_server.os.path.isabs', return_value=True) # Assume path is already absolute
    @patch('backend.mcp_server.chromadb_collection', MagicMock())
    @patch('backend.mcp_server.generate_query_embedding', MagicMock()) 
    def test_process_documents_successful_absolute_path(self, mock_isabs, mock_isdir, mock_process_docs):
        # Setup mock for absolute path
        mock_process_docs.return_value = {
            "status": "success", "processed_files": 1, "total_chunks_added": 10, "collection_total_items": 100
        }
        
        request = mcp_pb2.MCPProcessRequest(directory="/absolute/test/docs")
        response = self.servicer.ProcessDocuments(request, self.mock_context)

        mock_isabs.assert_called_once_with("/absolute/test/docs")
        mock_isdir.assert_called_once_with("/absolute/test/docs") # isdir called with the (now absolute) path
        mock_process_docs.assert_called_once_with(documents_dir="/absolute/test/docs") 
        
        self.assertEqual(response.status, "success")
        self.assertEqual(response.processed_files, 1)
        self.assertEqual(response.total_chunks_added, 10)
        self.assertEqual(response.collection_total_items, 100)
        self.mock_context.set_code.assert_not_called()


    @patch('backend.mcp_server.os.path.isdir', return_value=False) # Directory does not exist
    @patch('backend.mcp_server.os.path.isabs', return_value=True)  # Assume path is absolute for simplicity in this test
    @patch('backend.mcp_server.chromadb_collection', MagicMock())
    @patch('backend.mcp_server.generate_query_embedding', MagicMock())
    def test_process_documents_dir_not_found(self, mock_isabs, mock_isdir): # mock_isabs, mock_isdir must be params
        request = mcp_pb2.MCPProcessRequest(directory="/nonexistent/docs")
        self.servicer.ProcessDocuments(request, self.mock_context)
        
        mock_isabs.assert_called_once_with("/nonexistent/docs")
        mock_isdir.assert_called_once_with("/nonexistent/docs")
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        self.mock_context.set_details.assert_called_once()
        # Check that the detail message contains the path.
        self.assertIn("Directory not found: /nonexistent/docs", self.mock_context.set_details.call_args[0][0])


    @patch('backend.mcp_server.chromadb_collection', None) # Simulate component not available
    def test_process_documents_missing_chromadb(self):
        request = mcp_pb2.MCPProcessRequest(directory="/test/docs")
        self.servicer.ProcessDocuments(request, self.mock_context)
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.UNAVAILABLE)

    @patch('backend.mcp_server.process_docs_for_rag_internal', side_effect=Exception("Processing error"))
    @patch('backend.mcp_server.os.path.isdir', return_value=True) # Assume dir exists
    @patch('backend.mcp_server.os.path.isabs', return_value=True)  # Assume path is absolute
    @patch('backend.mcp_server.chromadb_collection', MagicMock())
    @patch('backend.mcp_server.generate_query_embedding', MagicMock())
    def test_process_documents_processing_error(self, mock_isabs, mock_isdir, mock_process_docs): # mock_isabs, mock_isdir, mock_process_docs must be params
        request = mcp_pb2.MCPProcessRequest(directory="/test/docs")
        self.servicer.ProcessDocuments(request, self.mock_context)
        
        mock_process_docs.assert_called_once_with(documents_dir="/test/docs")
        self.mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        self.mock_context.set_details.assert_called_with("An unexpected error occurred during document processing: Processing error")

if __name__ == '__main__':
    unittest.main()
