import json # For serializing Figma data for the prompt

# Import functionalities from other backend modules
from .figma_retriever import get_figma_nodes_info
from .vector_store import query_documents
from .llm_generator import generate_react_code_from_prompt
# from . import config # Not strictly needed if sub-modules handle their own config
import re # For regular expressions, e.g., in PascalCase conversion

def generate_react_for_figma_node(file_key: str, node_id: str) -> dict:
    """
    Orchestrates the RAG process to generate a React component for a given Figma node.

    Args:
        file_key (str): The Figma file key.
        node_id (str): The ID of the Figma node to generate code for.

    Returns:
        dict: A dictionary containing the generated React code and supporting information,
              or an error message.
    """
    # a. Fetch Figma Data
    print(f"Step 1: Fetching Figma data for file_key='{file_key}', node_id='{node_id}'...")
    try:
        figma_data_response = get_figma_nodes_info(file_key=file_key, node_ids=[node_id])
        if "nodes" not in figma_data_response or node_id not in figma_data_response["nodes"]:
            error_msg = f"Figma node '{node_id}' not found in response for file '{file_key}'."
            print(error_msg)
            return {"error": error_msg, "details": figma_data_response.get("err", "Node not found or access issue.")}
        
        # Extract the actual node data for the requested node_id
        # The structure is response -> "nodes" -> node_id -> "document"
        figma_node_info = figma_data_response["nodes"][node_id]
        if not figma_node_info or "document" not in figma_node_info:
            error_msg = f"Figma node data for '{node_id}' is incomplete or missing 'document' field."
            print(error_msg)
            return {"error": error_msg, "details": "Node data structure is not as expected."}
        
        figma_node_data = figma_node_info["document"] # This is the actual design data for the node
        figma_component_name = figma_node_data.get('name', 'UnnamedFigmaNode')
        print(f"Successfully fetched Figma data for: {figma_component_name} (Type: {figma_node_data.get('type')})")

    except Exception as e: # Covers HTTPException from figma_retriever or other issues
        error_msg = f"Error fetching Figma data for node '{node_id}': {str(e)}"
        print(error_msg)
        return {"error": error_msg, "details": str(e)}

    # b. Formulate Query for Vector Store
    node_type = figma_node_data.get('type', 'component')
    query_text = f"React component for a {node_type.lower()} similar to '{figma_component_name}'"
    # Future refinement: Add more descriptive terms from figma_node_data if available and useful.
    # e.g., "button", "card", "input field with label"
    # For now, type and name are good starting points.
    print(f"Step 2: Formulated query for vector store: '{query_text}'")

    # c. Retrieve Relevant Components
    print(f"Step 3: Retrieving relevant components from vector store (n_results=1)...")
    retrieved_docs = query_documents(query_text, n_results=1)
    retrieved_code_context = "No specific existing component found as direct context."
    retrieved_context_source = None

    if retrieved_docs and retrieved_docs[0] and "content" in retrieved_docs[0]:
        retrieved_code_context = retrieved_docs[0]['content']
        retrieved_context_source = retrieved_docs[0].get('metadata', {}).get('filename')
        print(f"Found relevant component: {retrieved_context_source if retrieved_context_source else 'Unknown source'}")
    else:
        print("No relevant components found in vector store.")

    # d. Construct the LLM Prompt
    # Basic serialization of some Figma properties. This needs significant refinement
    # to be truly useful for the LLM (e.g., extracting meaningful style info).
    figma_props_for_prompt = {
        "name": figma_node_data.get('name', 'N/A'),
        "type": figma_node_data.get('type', 'N/A'),
        "absoluteBoundingBox": figma_node_data.get('absoluteBoundingBox', {}),
        "fills": figma_node_data.get('fills', []), # Example style property
        "strokes": figma_node_data.get('strokes', []), # Example style property
        "effects": figma_node_data.get('effects', []), # Example style property
        # Add other properties as deemed useful: e.g., children structure summary, text content if any.
    }
    figma_props_str = json.dumps(figma_props_for_prompt, indent=2)
    
    # Sanitize Figma component name for use in function/component naming
    # Remove spaces and special characters, ensure PascalCase
    base_name = figma_node_data.get('name', 'MyFigmaComponent')
    # Remove non-alphanumeric characters (except underscore if desired) then capitalize parts
    # A simple approach: remove non-alphanumeric, then ensure first letter is capitalized.
    # For true PascalCase from "some name here", it's more complex.
    # For now, a simplified sanitization:
    sanitized_name_parts = [part.capitalize() for part in re.sub(r'[^a-zA-Z0-9]+', ' ', base_name).split(' ')]
    pascal_case_name = "".join(sanitized_name_parts)
    if not pascal_case_name: pascal_case_name = "GeneratedComponent" # Fallback

    prompt = f"""
You are an expert React developer tasked with generating a new React functional component based on Figma design specifications and potentially similar existing components.

Your goal is to create a single, complete, and valid JSX file content for the new component.

**1. Figma Design Details:**
The component to be generated is based on the following Figma node data:
```json
{figma_props_str}
```

**2. Potentially Relevant Existing React Component (Context):**
If this existing component is structurally similar and a good starting point, adapt it. Otherwise, create a new structure based on the Figma details.
```jsx
{retrieved_code_context}
```

**3. Task & Instructions:**
*   Generate a new React functional component named `{pascal_case_name}`.
*   The component should primarily implement the design and properties described in the **Figma Design Details**.
*   Use the **Potentially Relevant Existing React Component** as a reference for structure or utility functions if it's helpful and aligns with the Figma design. Do not simply copy it if it contradicts the Figma specs.
*   Ensure all necessary React imports (e.g., `import React from 'react';`) are included.
*   Props for the new component should be derived from the Figma properties (e.g., text content, children, event handlers if implied). For instance, if the Figma component has text, expect a `label` or `text` prop. If it's a container, expect `children`.
*   Apply styling to match the Figma design. For simplicity in this exercise, you can use inline styles (via the `style` prop in JSX elements), but try to make them reflect the `fills`, `strokes`, `absoluteBoundingBox` (for dimensions, though absolute positioning is often avoided in reusable components unless specified), and other relevant Figma properties.
*   If the Figma component has children (nested layers), the generated React component should also support rendering children props or have a similar nested structure.

**Output Format:**
Respond with *only* the React component code as a single, valid JSX string.
Do NOT include any explanations, comments outside the code, or markdown formatting like \`\`\`jsx ... \`\`\` around the code block.
The component should be exportable as `export default {pascal_case_name};`.
"""
    print(f"Step 4: Constructed LLM prompt for component '{pascal_case_name}'.")
    # print(f"--- Prompt Start ---\n{prompt[:300]}...\n--- Prompt End ---") # For brevity

    # e. Call LLM for Code Generation
    print(f"Step 5: Calling LLM to generate React code for '{pascal_case_name}'...")
    generated_code = generate_react_code_from_prompt(
        prompt=prompt,
        figma_component_name=pascal_case_name # Pass the intended final name
    )
    print(f"Code generation complete for '{pascal_case_name}'.")

    # f. Return Result
    result = {
        "figma_node_id": node_id,
        "figma_component_name": figma_component_name, # Original Figma name
        "generated_component_name": pascal_case_name, # Name used in generated code
        "query_used_for_retrieval": query_text,
        "retrieved_context_source": retrieved_context_source,
        "retrieved_context_code_snippet": retrieved_code_context[:300] + "..." if retrieved_code_context else None, # Snippet for brevity
        "llm_prompt_snippet": prompt[:400] + "...", # Snippet for brevity
        "generated_react_code": generated_code
    }
    print("Step 6: Result prepared.")
    return result

if __name__ == '__main__':
    print("\n--- Testing React Generator Service ---")
    
    # Test with dummy Figma data (assuming figma_retriever will return dummy data
    # if FIGMA_API_KEY is not set, which is the case in CI/testing).
    # The dummy node "1:1" (Frame1) from figma_retriever.py should be used.
    test_file_key = "dummy_figma_file_key"
    test_node_id = "1:1" 

    print(f"\nAttempting to generate React component for Figma node: file='{test_file_key}', node='{test_node_id}'")
    
    # The sub-modules (figma_retriever, vector_store, llm_generator) should handle
    # their own states regarding API keys or model loading and return dummy/default data if needed.
    
    generation_result = generate_react_for_figma_node(file_key=test_file_key, node_id=test_node_id)

    print("\n--- Generation Result ---")
    # Pretty print the main parts of the result, keeping code snippets potentially long
    if "error" in generation_result:
        print(f"Error: {generation_result['error']}")
        if "details" in generation_result:
            print(f"Details: {generation_result['details']}")
    else:
        print(f"Figma Node ID: {generation_result.get('figma_node_id')}")
        print(f"Figma Component Name: {generation_result.get('figma_component_name')}")
        print(f"Generated Component Name: {generation_result.get('generated_component_name')}")
        print(f"Query for Retrieval: {generation_result.get('query_used_for_retrieval')}")
        print(f"Retrieved Context Source: {generation_result.get('retrieved_context_source', 'N/A')}")
        # print(f"Retrieved Context Snippet: {generation_result.get('retrieved_context_code_snippet', 'N/A')}")
        # print(f"LLM Prompt Snippet: {generation_result.get('llm_prompt_snippet', 'N/A')}")
        print("\n--- Generated React Code ---")
        print(generation_result.get('generated_react_code', 'No code generated.'))
        print("--- End of Generated React Code ---")

    print("\n--- React Generator Service Test Complete ---")
