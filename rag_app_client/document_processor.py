import os # For interacting with the file system, e.g., path operations.
from sentence_transformers import SentenceTransformer # For generating text embeddings.
from PyPDF2 import PdfReader # For extracting text from PDF files.
from docx import Document as DocxDocument # For extracting text from .docx files.

# --- Sentence Transformer Model Initialization ---
# Load a pre-trained sentence transformer model.
# This model will be used to convert text chunks into numerical embeddings.
# 'all-MiniLM-L6-v2' is a good starting model: fast and relatively small, with decent performance.
# The model is loaded once when this module is imported to avoid reloading on each use.
# This will download the model from Hugging Face Hub on first use if not already cached.
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    # If model loading fails (e.g., network issue, resource constraints), print an error
    # and set the model to None. Functions relying on the model should handle this.
    print(f"Error loading SentenceTransformer model: {e}")
    print("Embedding generation will not be available.")
    model = None

# --- Document Loading Functions ---

def load_txt(file_path: str) -> str:
    """
    Loads text content from a .txt file.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        str: The extracted text content from the file. Returns an empty string if loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading TXT file {file_path}: {e}")
        return ""

def load_pdf(file_path: str) -> str:
    """
    Extracts text content from all pages of a .pdf file.

    Args:
        file_path (str): The path to the .pdf file.

    Returns:
        str: The concatenated text content from all pages. Returns an empty string if extraction fails
             or if the required library (PyPDF2) is not correctly installed.
    """
    text = ""
    try:
        with open(file_path, 'rb') as f: # Open PDF in binary read mode
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted from the page
                    text += page_text
        return text
    except Exception as e:
        # Common issues: PyPDF2 not installed, encrypted PDF, corrupted file.
        print(f"Error loading PDF file {file_path}: {e}. Make sure PyPDF2 is installed and file is valid.")
        return ""

def load_docx(file_path: str) -> str:
    """
    Extracts text content from all paragraphs of a .docx file.

    Args:
        file_path (str): The path to the .docx file.

    Returns:
        str: The concatenated text content from all paragraphs. Returns an empty string if extraction fails
             or if the required library (python-docx) is not correctly installed.
    """
    text = ""
    try:
        doc = DocxDocument(file_path) # Load the .docx file
        for para in doc.paragraphs: # Iterate through each paragraph in the document
            text += para.text + "\n" # Append paragraph text with a newline
        return text.strip() # Remove any trailing newline
    except Exception as e:
        # Common issues: python-docx not installed, corrupted file.
        print(f"Error loading DOCX file {file_path}: {e}. Make sure python-docx is installed and file is valid.")
        return ""

# --- Text Processing Functions ---

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Splits a given text into smaller chunks based on word count, with a specified overlap
    between consecutive chunks. This is useful for preparing text for embedding models
    that have input length limitations.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired number of words in each chunk.
        overlap (int): The number of words to overlap between consecutive chunks.
                       This helps maintain context across chunks.

    Returns:
        list[str]: A list of text chunks. Returns an empty list if the input text is empty.
    """
    if not text:
        return []

    words = text.split() # Split the text into words
    chunks = []

    # Iterate through the words, creating chunks with the specified size and overlap
    # The step is `chunk_size - overlap` to account for the overlapping words.
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size]) # Join words to form a chunk
        chunks.append(chunk)
    return chunks

def generate_embeddings(text_chunks: list[str]) -> list[list[float]]:
    """
    Generates numerical embeddings for a list of text chunks using the pre-loaded
    SentenceTransformer model.

    Args:
        text_chunks (list[str]): A list of text chunks for which to generate embeddings.

    Returns:
        list[list[float]]: A list of embeddings, where each embedding is a list of floats.
                           Returns an empty list if the model is not loaded, if input is empty,
                           or if an error occurs during embedding generation.
    """
    # Check if the SentenceTransformer model was loaded successfully.
    if model is None:
        print("Error: SentenceTransformer model is not loaded. Cannot generate embeddings.")
        return []

    if not text_chunks: # If there are no chunks, return an empty list.
        return []

    try:
        # Generate embeddings using the model's encode method.
        # `convert_to_tensor=False` ensures the output is a NumPy array or list of lists,
        # which is easier to handle for JSON serialization or direct use.
        embeddings = model.encode(text_chunks, convert_to_tensor=False)

        # Ensure embeddings are returned as a list of lists of floats.
        # The `encode` method might return a NumPy array, so convert it if necessary.
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    # This block executes only when the script is run directly (e.g., `python document_processor.py`).
    # It's useful for testing the functionalities of this module independently.
    print("\nTesting document processing functions...")

    # --- Test TXT Loading and Processing ---
    # Construct path to a sample TXT file (assuming it's in a subdirectory relative to this script)
    sample_txt_path = os.path.join(os.path.dirname(__file__), 'documents_for_rag', 'sample1.txt')

    # For robust testing, create a dummy file if it doesn't exist.
    if not os.path.exists(sample_txt_path):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'documents_for_rag'), exist_ok=True)
        with open(sample_txt_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample text file for testing the document_processor.py module. It contains several sentences.")
        print(f"Created dummy file for testing: {sample_txt_path}")

    txt_content = load_txt(sample_txt_path)
    if txt_content:
        print(f"\n--- TXT Content (first 100 chars): ---\n{txt_content[:100]}...")
        txt_chunks = chunk_text(txt_content, chunk_size=10) # Use smaller chunk size for testing
        print(f"\n--- TXT Chunks (first 3): ---\n{txt_chunks[:3]}")

        if model: # Check if the embedding model is loaded
            # Generate embeddings for the first few chunks for brevity
            txt_embeddings = generate_embeddings(txt_chunks[:1])
            if txt_embeddings:
                print(f"\n--- TXT Embedding (first chunk, first 5 dims): ---\n{txt_embeddings[0][:5]}...")
            else:
                print("\n--- TXT Embedding generation failed. ---")
        else:
            print("\n--- Skipping TXT embedding generation as model is not loaded. ---")
    else:
        print("\n--- TXT Loading Failed ---")

    # --- Test PDF Loading (Illustrative) ---
    # Note: PDF and DOCX testing might fail if their respective libraries (PyPDF2, python-docx)
    # were not installed correctly, e.g., due to environment constraints.
    # This section demonstrates how one might test them.

    # Path to a dummy PDF (you would need to create a sample.pdf for this to run)
    sample_pdf_path = os.path.join(os.path.dirname(__file__), 'documents_for_rag', 'sample.pdf')
    if not os.path.exists(sample_pdf_path):
        print(f"\nDummy PDF file {sample_pdf_path} not found. Skipping PDF test.")
    else:
        pdf_content = load_pdf(sample_pdf_path)
        if pdf_content:
            print(f"\n--- PDF Content (first 100 chars): ---\n{pdf_content[:100]}...")
            pdf_chunks = chunk_text(pdf_content, chunk_size=50)
            print(f"\n--- PDF Chunks (first 2): ---\n{pdf_chunks[:2]}")
        else:
            print("\n--- PDF Loading Failed (or file is empty/corrupt, or PyPDF2 is missing/failed). ---")

    # --- Test DOCX Loading (Illustrative) ---
    # Path to a dummy DOCX (you would need to create a sample.docx for this to run)
    sample_docx_path = os.path.join(os.path.dirname(__file__), 'documents_for_rag', 'sample.docx')
    if not os.path.exists(sample_docx_path):
         print(f"\nDummy DOCX file {sample_docx_path} not found. Skipping DOCX test.")
    else:
        docx_content = load_docx(sample_docx_path)
        if docx_content:
            print(f"\n--- DOCX Content (first 100 chars): ---\n{docx_content[:100]}...")
            docx_chunks = chunk_text(docx_content, chunk_size=50)
            print(f"\n--- DOCX Chunks (first 2): ---\n{docx_chunks[:2]}")
        else:
            print("\n--- DOCX Loading Failed (or file is empty/corrupt, or python-docx is missing/failed). ---")
