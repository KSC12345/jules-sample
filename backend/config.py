import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
FIGMA_API_KEY = os.getenv("FIGMA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Model Configuration
# Defaults to a common sentence transformer model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')

# ChromaDB Configuration
# Defaults for local development
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "react_components")

# It is recommended to set these variables in a .env file in the backend directory.
# See .env.example for a template.
# Example .env file content:
# FIGMA_API_KEY="your_figma_api_key_here"
# OPENAI_API_KEY="your_openai_api_key_here"
# Optional: Override default embedding model
# EMBEDDING_MODEL_NAME="another_sentence_transformer_model"
# Optional: Override default ChromaDB path
# CHROMA_DB_PATH="./custom_chroma_data"
# Optional: Override default Chroma collection name
# CHROMA_COLLECTION_NAME="custom_collection_name"
