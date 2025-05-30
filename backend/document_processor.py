import os # For interacting with the file system, e.g., path operations.
# SentenceTransformer will be initialized in vector_store.py using config
from PyPDF2 import PdfReader # For extracting text from PDF files.
from docx import Document as DocxDocument # For extracting text from .docx files.
import glob # For finding files matching a pattern

from . import config # Import config for accessing directory paths

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


def load_react_component_file(file_path: str) -> str:
    """
    Loads content from a React component file (.jsx, .js, .tsx, .ts).
    Currently, this is the same as loading a TXT file.
    Future enhancements could include parsing or specific validation.
    """
    return load_txt(file_path)


def load_and_chunk_react_components(components_dir: str = "backend/retrieved_components/") -> list[dict]:
    """
    Loads all React component files from the specified directory.
    Each file's content is treated as a single "chunk".

    Args:
        components_dir (str): The directory containing React component files.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary has
                    'content' (the file content) and 'metadata' (filename).
    """
    documents = []
    valid_extensions = ['.js', '.jsx', '.ts', '.tsx']

    # Ensure components_dir path is correct, potentially using config if it were defined there
    # For now, using the provided default or argument.

    for ext in valid_extensions:
        # Use glob to find all files with the current extension in the directory
        # The pattern `*` matches any characters, so `*` + `ext` matches all files ending with `ext`.
        # `recursive=True` could be used if components are in subdirectories, along with `**/*` pattern.
        # For now, assuming flat structure in components_dir.
        search_pattern = os.path.join(components_dir, f"*{ext}")
        for file_path in glob.glob(search_pattern):
            print(f"Processing component file: {file_path}")
            content = load_react_component_file(file_path)
            if content:
                documents.append({
                    "content": content,
                    "metadata": {"filename": os.path.basename(file_path)}
                })
            else:
                print(f"Could not load content from {file_path}")

    if not documents:
        print(f"No component files found or loaded in directory: {components_dir}")
        # Example: Check if the directory actually exists or has files
        if not os.path.exists(components_dir):
            print(f"Error: Components directory '{components_dir}' does not exist.")
        else:
            print(f"Directory '{components_dir}' exists but no files with extensions {valid_extensions} were found or loaded.")


    return documents


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

# generate_embeddings function is removed as it will be part of vector_store.py

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    print("\nTesting document processing functions...")

    # --- Test React Component Loading ---
    # The default path for components is 'backend/retrieved_components/'
    # Ensure this path is correct relative to where this script might be run from,
    # or provide an absolute path / correctly relative path if needed.
    # For example, if script is run from /app, 'backend/retrieved_components/' is correct.
    
    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the retrieved_components directory relative to the script's directory
    # This assumes 'retrieved_components' is a sibling to the directory containing this script,
    # which is not the case here.
    # The structure is /app/backend/document_processor.py and /app/backend/retrieved_components
    # So, the default "backend/retrieved_components/" is actually incorrect if script is run from /app/backend
    # It should be "./retrieved_components/" or "../backend/retrieved_components" if in a sub-folder of backend.
    # Let's assume it's run from /app for now, so "backend/retrieved_components/" is fine.
    # However, for robustness, using an absolute path or a path relative to a known root (like config.BASE_DIR if we had one)
    # is better. The default argument `backend/retrieved_components/` is relative to the CWD.

    # For testing from `python backend/document_processor.py` inside `/app`
    # The CWD would be `/app`. So `backend/retrieved_components` is the correct path.

    print("\n--- Testing React Component Loading ---")
    # Use the default path specified in the function definition
    react_components_path = "backend/retrieved_components/"

    # Check if the directory exists to provide better feedback
    if not os.path.isdir(react_components_path):
        print(f"Error: Test components directory '{react_components_path}' not found from CWD '{os.getcwd()}'.")
        print("Please ensure the path is correct or the script is run from the project root (/app).")
        # Attempt to use a path relative to this script file for robustness in testing.
        # This assumes 'retrieved_components' is in the same directory as 'document_processor.py'.
        # This is incorrect for the current structure.
        # For now, we'll rely on the default path and the user running from /app.
    
    loaded_components = load_and_chunk_react_components() # Uses default path

    if loaded_components:
        print(f"\nSuccessfully loaded {len(loaded_components)} component(s).")
        for i, doc in enumerate(loaded_components):
            print(f"\n--- Component {i+1}: {doc['metadata']['filename']} ---")
            print(f"Content (first 100 chars): {doc['content'][:100]}...")
    else:
        print("\n--- No React components loaded. Check paths and file extensions. ---")
        print(f"Searched in: {os.path.abspath(react_components_path)}")


    # Remove old test for TXT, PDF, DOCX to keep focus on React components for now
    # If needed, they can be added back or tested separately.
    print("\n--- Other document type tests (TXT, PDF, DOCX) were removed for this example. ---")
