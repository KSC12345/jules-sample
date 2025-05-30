import pytest
import os
import shutil
import tempfile

# Add project root to sys.path to allow imports like `from backend.document_processor...`
# This is often needed when running tests directly without full package installation.
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__)) # tests directory
# project_root = os.path.dirname(current_dir) # backend directory
# app_root = os.path.dirname(project_root) # /app directory
# if app_root not in sys.path:
#    sys.path.insert(0, app_root)
# The above logic might be needed if running `python backend/tests/test_document_processor.py`
# However, with `python -m pytest backend/tests`, pytest handles path discovery better.
# For now, let's rely on pytest's path handling or ensure PYTHONPATH is set correctly in the environment.
# If imports fail, this sys.path manipulation is the first thing to check.

from backend.document_processor import load_and_chunk_react_components

# --- Fixtures ---

@pytest.fixture
def empty_dir(tmp_path):
    """Creates an empty temporary directory using pytest's tmp_path fixture."""
    # tmp_path is a pathlib.Path object provided by pytest for temporary file system resources.
    # It's automatically managed and cleaned up by pytest.
    test_dir = tmp_path / "empty_test_dir"
    test_dir.mkdir()
    print(f"Created empty_dir: {test_dir}")
    return str(test_dir) # Return path as string, as os.path.join might be used internally

@pytest.fixture
def valid_components_dir(tmp_path):
    """Creates a temporary directory with a couple of valid React component files."""
    test_dir = tmp_path / "valid_components_test_dir"
    test_dir.mkdir()
    print(f"Created valid_components_dir: {test_dir}")

    button_jsx_content = """
    import React from 'react';
    function Button({ label }) {
      return <button>{label}</button>;
    }
    export default Button;
    """
    with open(test_dir / "Button.jsx", "w") as f:
        f.write(button_jsx_content)

    card_tsx_content = """
    import React from 'react';
    interface CardProps { title: string; children: React.ReactNode; }
    const Card: React.FC<CardProps> = ({ title, children }) => {
      return <div data-testid="card"><h3>{title}</h3><div>{children}</div></div>;
    }
    export default Card;
    """
    with open(test_dir / "Card.tsx", "w") as f:
        f.write(card_tsx_content)
        
    return str(test_dir)

@pytest.fixture
def mixed_files_dir(tmp_path):
    """Creates a temporary directory with a mix of React and non-React files."""
    test_dir = tmp_path / "mixed_files_test_dir"
    test_dir.mkdir()
    print(f"Created mixed_files_dir: {test_dir}")

    # Valid files
    (test_dir / "Component1.js").write_text("export default () => <p>Component1 JS</p>;")
    (test_dir / "Component2.jsx").write_text("export default () => <p>Component2 JSX</p>;")
    (test_dir / "Component3.ts").write_text("export default () => <p>Component3 TS</p>;") # Simplified for test
    (test_dir / "Component4.tsx").write_text("export default () => <p>Component4 TSX</p>;") # Simplified for test

    # Invalid files (should be ignored)
    (test_dir / "notes.txt").write_text("This is a text file.")
    (test_dir / "script.py").write_text("print('This is a python script')")
    (test_dir / "README.md").write_text("# Markdown file")
    
    # Subdirectory (should be ignored by current non-recursive implementation)
    sub_dir = test_dir / "subdirectory"
    sub_dir.mkdir()
    (sub_dir / "NestedComponent.jsx").write_text("export default () => <p>Nested</p>;")

    return str(test_dir)

# --- Test Cases ---

def test_load_empty_dir(empty_dir):
    """Test loading from an empty directory."""
    print(f"Testing with empty_dir: {empty_dir}")
    documents = load_and_chunk_react_components(components_dir=empty_dir)
    assert len(documents) == 0, "Should return an empty list for an empty directory."

def test_load_valid_components(valid_components_dir):
    """Test loading from a directory with valid React components."""
    print(f"Testing with valid_components_dir: {valid_components_dir}")
    documents = load_and_chunk_react_components(components_dir=valid_components_dir)
    assert len(documents) == 2, "Should load two component files."

    filenames = sorted([doc['metadata']['filename'] for doc in documents])
    assert filenames == ["Button.jsx", "Card.tsx"], "Filenames should be correctly extracted."

    button_doc = next((doc for doc in documents if doc['metadata']['filename'] == "Button.jsx"), None)
    assert button_doc is not None, "Button.jsx document should be found."
    assert "function Button({ label })" in button_doc['content'], "Content of Button.jsx seems incorrect."
    
    card_doc = next((doc for doc in documents if doc['metadata']['filename'] == "Card.tsx"), None)
    assert card_doc is not None, "Card.tsx document should be found."
    assert "interface CardProps" in card_doc['content'], "Content of Card.tsx seems incorrect."


def test_load_mixed_files(mixed_files_dir):
    """Test loading from a directory with mixed file types."""
    print(f"Testing with mixed_files_dir: {mixed_files_dir}")
    documents = load_and_chunk_react_components(components_dir=mixed_files_dir)
    
    assert len(documents) == 4, "Should only load files with .js, .jsx, .ts, .tsx extensions from the top level."

    valid_filenames = {"Component1.js", "Component2.jsx", "Component3.ts", "Component4.tsx"}
    loaded_filenames = {doc['metadata']['filename'] for doc in documents}
    assert loaded_filenames == valid_filenames, "Only files with valid extensions should be loaded."

    for doc in documents:
        assert isinstance(doc['content'], str) and len(doc['content']) > 0, "Document content should be a non-empty string."
        assert doc['metadata']['filename'] in valid_filenames, "Metadata filename should be one of the valid files."

def test_load_non_existent_dir():
    """Test loading from a non-existent directory."""
    # Use a unique name within pytest's tmp_path for this test, though it won't be created.
    # tmp_path itself is a good base if we were to use pathlib more directly.
    # For this test, just need a path that almost certainly doesn't exist.
    non_existent_path = os.path.join(str(tempfile.gettempdir()), "unique_non_existent_dir_12345") # Use tempfile for a more conventional temp area
    
    # Ensure it really doesn't exist from a previous failed run if using a fixed path.
    if os.path.exists(non_existent_path):
        shutil.rmtree(non_existent_path) 

    print(f"Testing with non_existent_dir: {non_existent_path}")
    # The function itself prints an error to console, test asserts it returns empty list.
    documents = load_and_chunk_react_components(components_dir=non_existent_path)
    assert len(documents) == 0, "Should return an empty list for a non-existent directory."

# Note on sys.path for running tests:
# If running tests using `python -m pytest backend/tests`, pytest typically handles
# path discovery well, and explicit sys.path manipulation might not be needed at the top of this file.
# The `from backend.document_processor ...` import relies on 'backend' being discoverable.
# If issues arise, ensure the tests are run from the '/app' directory or that PYTHONPATH includes '/app'.
# The fixtures now use pytest's `tmp_path` which is preferred over `tempfile.mkdtemp()` for pytest tests.
# `tmp_path` returns a `pathlib.Path` object. Converted to `str` for `load_and_chunk_react_components`
# if it expects string paths (os.path.join internally will handle Path objects too in modern Python).
