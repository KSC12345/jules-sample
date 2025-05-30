import openai
import os
import re # Import the 're' module for regular expressions

# Attempt to import OPENAI_API_KEY from backend.config
try:
    from . import config as backend_config
    OPENAI_API_KEY = backend_config.OPENAI_API_KEY
except ImportError:
    # Fallback for direct execution or if backend.config is not found initially.
    print("Could not import backend.config. Attempting to load OPENAI_API_KEY directly from environment.")
    from dotenv import load_dotenv
    load_dotenv() # Load .env file if present
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client if API key is available
if OPENAI_API_KEY:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None
else:
    print("Warning: OPENAI_API_KEY not found. OpenAI client not initialized. Will use dummy responses for generation.")
    client = None

# The Hugging Face transformers-based generator_pipeline and related code (model, tokenizer, generate_response_from_context)
# are removed as per the subtask to focus on OpenAI for code generation.
# If the RAG chat functionality using a local LLM (like distilgpt2) is still needed,
# it would either need to be reinstated or moved to a separate module.
# For this subtask, we assume it's being replaced or handled elsewhere.
# Global variables for Hugging Face pipeline (commented out or removed):
# MODEL_NAME = "distilgpt2" 
# tokenizer = None
# model = None
# generator_pipeline = None
# ... (Hugging Face model/tokenizer loading code removed) ...
# def generate_response_from_context(...) function removed ...


def generate_react_code_from_prompt(prompt: str, figma_component_name: str = "UnknownComponent") -> str:
    """
    Generates React component code based on a given prompt using OpenAI's API,
    or returns a dummy component if the API key is not configured or an error occurs.

    Args:
        prompt (str): The detailed prompt describing the React component to be generated.
        figma_component_name (str): The name of the Figma component, used for dummy generation.
                                    Defaults to "UnknownComponent".

    Returns:
        str: The generated React component code as a string, or a dummy component string.
    """
    component_name_for_dummy = figma_component_name if figma_component_name else "GeneratedComponent"
    # Sanitize the component name for use in function names
    component_name_safe = re.sub(r'[^a-zA-Z0-9_]', '', component_name_for_dummy)
    if not component_name_safe: # If sanitization results in empty string, use a default
        component_name_safe = "GeneratedComponent"
    
    prompt_snippet = prompt[:500] + ('...' if len(prompt) > 500 else '')
    # Escape characters that might break JS template literals or JSX content, then remove outer quotes if json.dumps added them.
    # This is a simple approach; a more robust HTML/JS escaping might be needed for arbitrary prompts.
    # For now, focusing on braces and backticks often found in code snippets within prompts.
    prompt_snippet_escaped_for_js_template_literal = prompt_snippet.replace('`', '\\`').replace('${', '\\${')


    # Using .format() for the template string to avoid complex f-string escaping issues.
    # All literal curly braces for JS/JSX must be doubled (e.g., {{ or }}).
    # Placeholders for .format() are single (e.g., {py_var_name}).
    dummy_jsx_template_str = """// Dummy component, LLM not called, API key missing, or error during generation.
// Component Name: {py_component_name_for_dummy}
import React from 'react';

function DummyFigmaComponent_{py_component_name_safe}(props) {{
  const figmaComponentName = props.figmaComponentName || "{py_component_name_for_dummy}";
  return (
    <div style={{{{ padding: '20px', border: '2px dashed #ccc', backgroundColor: '#f9f9f9', borderRadius: '8px' }}}}>
      <h3 style={{{{ color: '#555' }}}}>Placeholder for: {{figmaComponentName}}</h3>
      <p style={{{{ color: '#777' }}}}>This is a dummy component. If you see this, it means the AI code generation did not run.</p>
      <p style={{{{ color: '#777' }}}}>The actual component would be generated based on Figma properties and retrieved similar components.</p>
      <details>
        <summary style={{{{ cursor: 'pointer', color: '#007bff' }}}}>Prompt Details (for debugging)</summary>
        <pre style={{{{ backgroundColor: '#eee', padding: '10px', borderRadius: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}}}>
          {{`{py_prompt_snippet_escaped}`}}
        </pre>
      </details>
    </div>
  );
}}

export default DummyFigmaComponent_{py_component_name_safe};
"""
    # Prepare the formatted dummy code once.
    current_dummy_code = dummy_jsx_template_str.format(
        py_component_name_for_dummy=component_name_for_dummy,
        py_component_name_safe=component_name_safe,
        py_prompt_snippet_escaped=prompt_snippet_escaped_for_js_template_literal
    )

    if not client:
        print(f"Warning: OpenAI client not initialized (API key likely missing). Returning dummy React component for '{component_name_for_dummy}'.")
        return current_dummy_code

    try:
        print(f"Attempting to generate React code for '{figma_component_name}' using OpenAI API...")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # A good default, consider gpt-4 for more complex tasks if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates React component code based on detailed specifications. Output only the React component code as a single JSX string. Do not include any explanatory text before or after the code block. Ensure the component is functional and uses props as described."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000, # Increased to allow for potentially larger components
            temperature=0.2, # Lower temperature for more deterministic code generation
            # top_p=1.0,
            # frequency_penalty=0.0,
            # presence_penalty=0.0
        )
        
        generated_code = completion.choices[0].message.content.strip()
        
        # Basic check to see if the output seems like code (e.g., contains "import React")
        if "import React" not in generated_code and "function" not in generated_code:
            print("Warning: Generated content doesn't look like a React component. It might be an error message or unexpected output from LLM.")
            # Fallback to dummy if output is suspicious
            # return dummy_jsx_template.format(prompt=prompt) # Optionally, return dummy on suspicious output
        
        print(f"Successfully generated React code for '{figma_component_name}'.")
        # Clean up potential markdown code block delimiters
        if generated_code.startswith("```jsx"):
            generated_code = generated_code[len("```jsx"):].strip()
        if generated_code.startswith("```javascript"):
            generated_code = generated_code[len("```javascript"):].strip()
        if generated_code.startswith("```"):
             generated_code = generated_code[len("```"):].strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-len("```")].strip()
            
        return generated_code

    except openai.APIError as e:
        print(f"OpenAI API error when generating code for '{figma_component_name}': {e}")
        print("Returning dummy component instead.")
        return current_dummy_code
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI API call for '{figma_component_name}': {e}")
        print("Returning dummy component instead.")
        return current_dummy_code

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    print("\n--- Testing OpenAI React Code Generation ---")

    sample_figma_data = """
    Figma Component Name: PrimaryButton
    Properties:
    - Variant: Primary, Secondary, Destructive
    - Size: Small, Medium, Large
    - Text content: "Submit"
    - Icon: Optional, left-aligned
    - Colors: Primary Blue for Primary variant, Red for Destructive.
    - Dimensions: height 40px, padding 16px.
    """

    sample_retrieved_code = """
    // Similar Button component (example)
    import React from 'react';
    const BaseButton = ({ children, style, ...props }) => (
      <button style={{ ...style }} {...props}>{children}</button>
    );
    export default BaseButton;
    """

    sample_prompt = f"""
    Generate a React functional component using JSX based on the following specifications.
    The component should be a single file, including necessary imports like React.
    It should be well-structured and follow common React best practices.

    Figma Component Details:
    ```json
    {sample_figma_data}
    ```

    Retrieved Similar Component Code (for reference, adapt if useful):
    ```jsx
    {sample_retrieved_code}
    ```

    Instructions for the new component:
    1. Name the component `NewPrimaryButton`.
    2. It should accept props: `variant` (string: "primary", "secondary", "destructive"), `size` (string: "small", "medium", "large"), `label` (string), `onClick` (function), `icon` (optional ReactNode).
    3. Implement basic styling directly inline using the `style` prop for simplicity for now.
       - Primary: blue background, white text.
       - Secondary: gray background, black text.
       - Destructive: red background, white text.
       - Small: padding 8px, fontSize 12px.
       - Medium: padding 12px, fontSize 14px.
       - Large: padding 16px, fontSize 16px.
    4. If an `icon` prop is provided, display it to the left of the label.
    5. The component should be exportable as default.
    Ensure the output is ONLY the React component code string.
    """

    test_component_name = "NewPrimaryButtonTest"
    print(f"\nGenerating component: {test_component_name}")
    
    if client:
        print("Attempting LIVE OpenAI API call...")
    else:
        print("OpenAI client not initialized. Expecting DUMMY response.")

    generated_code = generate_react_code_from_prompt(sample_prompt, test_component_name)
    
    print(f"\n--- Generated Code for {test_component_name} ---")
    print(generated_code)
    print("--- End of Generated Code ---")

    if "DummyFigmaComponent" in generated_code:
        print("\nTest Result: DUMMY component was returned.")
    elif "NewPrimaryButton" in generated_code : # Check for expected component name from prompt
        print("\nTest Result: Potentially LIVE component was returned (or a very good dummy).")
    else:
        print("\nTest Result: Output received, review manually.")
        
    if not OPENAI_API_KEY:
        print("\n(Reminder: OPENAI_API_KEY was not set for this test run.)")
    else:
        print(f"\n(OPENAI_API_KEY is set, first 5 chars: {OPENAI_API_KEY[:5]}...)")

    print("\n--- LLM Generator Test Complete ---")
