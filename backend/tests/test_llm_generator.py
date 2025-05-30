import pytest
import openai # For openai.APIError
from unittest.mock import patch, MagicMock
import importlib # For reloading modules

# Ensure backend modules are discoverable
from backend.llm_generator import generate_react_code_from_prompt

# --- Fixtures ---

@pytest.fixture
def mock_openai_chat_completions_create():
    """Mocks openai.OpenAI().chat.completions.create method."""
    # We need to mock the method on an *instance* of the client.
    # If 'client' is a global in llm_generator, we can patch 'backend.llm_generator.client.chat.completions.create'.
    # Or, if client is instantiated per call (less likely for OpenAI client), it's harder.
    # llm_generator.py initializes 'client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None'
    # So, we patch the 'create' method that would be called on this 'client' instance.
    with patch('openai.resources.chat.completions.Completions.create') as mock_create:
        yield mock_create

# The reset_llm_generator_client_state fixture is removed.
# We will handle OPENAI_API_KEY patching and module reloading directly in tests that need it.

# --- Test Cases ---

def test_generate_api_key_missing(monkeypatch, mock_openai_chat_completions_create):
    """Test behavior when OPENAI_API_KEY is not set."""
    # Patch backend.config where llm_generator reads the key from.
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', None)
    
    # Reload llm_generator to make it re-evaluate its OPENAI_API_KEY and client
    import backend.llm_generator
    importlib.reload(backend.llm_generator)

    prompt = "A simple prompt"
    component_name = "TestComponentMissingKey"
    
    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    
    assert f"// Component Name: {component_name}" in result
    assert "DummyFigmaComponent_TestComponentMissingKey" in result 
    assert "API key missing" in result # Part of the comment in dummy JSX
    mock_openai_chat_completions_create.assert_not_called()

def test_generate_success(monkeypatch, mock_openai_chat_completions_create):
    """Test successful code generation via OpenAI API mock."""
    # Patch backend.config's OPENAI_API_KEY
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', "fake_openai_api_key")
    
    # Reload llm_generator to re-initialize its client with the patched key
    import backend.llm_generator 
    importlib.reload(backend.llm_generator)

    prompt = "Create a blue button."
    component_name = "BlueButton"
    expected_code = "import React from 'react';\n\nfunction BlueButton() { return <button style={{backgroundColor: 'blue'}}>Blue Button</button>; }\n\nexport default BlueButton;"

    # Mock the OpenAI API response structure
    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = expected_code
    
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]
    mock_openai_chat_completions_create.return_value = mock_completion_response
    
    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    
    assert result == expected_code
    mock_openai_chat_completions_create.assert_called_once()
    # We can also inspect the call arguments if needed, e.g.:
    # args, kwargs = mock_openai_chat_completions_create.call_args
    # assert kwargs['model'] == "gpt-3.5-turbo"
    # assert kwargs['messages'][1]['content'] == prompt


def test_generate_api_error(monkeypatch, mock_openai_chat_completions_create):
    """Test API error handling, expecting fallback to dummy component."""
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', "fake_openai_api_key")
    import backend.llm_generator 
    importlib.reload(backend.llm_generator)

    prompt = "Prompt that causes API error."
    component_name = "ErrorComponent"
    
    mock_openai_chat_completions_create.side_effect = openai.APIError("Simulated API Error", request=None, body=None)
    
    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    
    assert f"// Component Name: {component_name}" in result
    assert "DummyFigmaComponent_ErrorComponent" in result
    assert "error during generation" in result # Part of dummy comment
    mock_openai_chat_completions_create.assert_called_once()

def test_generate_code_cleanup_jsx_markdown(monkeypatch, mock_openai_chat_completions_create):
    """Test cleanup of markdown backticks for JSX."""
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', "fake_openai_api_key")
    import backend.llm_generator
    importlib.reload(backend.llm_generator)

    prompt = "Create component with backticks."
    component_name = "MarkdownCleanupJsx"
    raw_code = "```jsx\nimport React from 'react';\nfunction MyTestComponent() { return <p>Test</p>; }\nexport default MyTestComponent;\n```"
    expected_cleaned_code = "import React from 'react';\nfunction MyTestComponent() { return <p>Test</p>; }\nexport default MyTestComponent;"

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = raw_code
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]
    mock_openai_chat_completions_create.return_value = mock_completion_response
    
    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    assert result == expected_cleaned_code

def test_generate_code_cleanup_generic_markdown(monkeypatch, mock_openai_chat_completions_create):
    """Test cleanup of generic markdown backticks."""
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', "fake_openai_api_key")
    import backend.llm_generator
    importlib.reload(backend.llm_generator)

    prompt = "Create component with backticks."
    component_name = "MarkdownCleanupGeneric"
    raw_code = "```\nimport React from 'react';\nfunction MyTestComponent() { return <p>Test</p>; }\nexport default MyTestComponent;\n```"
    expected_cleaned_code = "import React from 'react';\nfunction MyTestComponent() { return <p>Test</p>; }\nexport default MyTestComponent;"
    
    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = raw_code
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]
    mock_openai_chat_completions_create.return_value = mock_completion_response

    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    assert result == expected_cleaned_code

def test_generate_code_no_markdown(monkeypatch, mock_openai_chat_completions_create):
    """Test code without markdown backticks is unchanged."""
    monkeypatch.setattr('backend.config.OPENAI_API_KEY', "fake_openai_api_key")
    import backend.llm_generator
    importlib.reload(backend.llm_generator)


    prompt = "Create component no backticks."
    component_name = "NoMarkdown"
    clean_code = "import React from 'react';\nfunction MyTestComponent() { return <p>Test</p>; }\nexport default MyTestComponent;"

    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = clean_code
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]
    mock_openai_chat_completions_create.return_value = mock_completion_response

    result = generate_react_code_from_prompt(prompt, figma_component_name=component_name)
    assert result == clean_code
