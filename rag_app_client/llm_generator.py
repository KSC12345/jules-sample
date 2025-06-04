from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
# Specify the Hugging Face model name for the language model.
# Using a smaller model like "distilgpt2" is advisable for environments with limited resources
# (CPU, memory, disk space for model download).
MODEL_NAME = "distilgpt2"

# --- Global Variables ---
# These variables will hold the loaded tokenizer, model, and text generation pipeline.
# They are loaded once when the module is imported to avoid reloading on every API call,
# which would be very inefficient.
tokenizer = None
model = None
generator_pipeline = None

# --- Model and Tokenizer Initialization ---
try:
    # Attempt to load the tokenizer and model from Hugging Face Hub.
    # This will download the model files if they are not already cached locally.
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer for {MODEL_NAME} loaded successfully.")

    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print(f"Model {MODEL_NAME} loaded successfully.")

    # Set pad_token_id if it's not already set. This is a common requirement for some
    # GPT-2 based models when using them in a pipeline for tasks like text generation.
    # If pad_token is None, it's often set to eos_token (end-of-sentence token).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Also update the model's configuration to reflect this.
        model.config.pad_token_id = model.config.eos_token_id

    # Create a text generation pipeline using the loaded model and tokenizer.
    # The pipeline simplifies the process of using the model for text generation.
    generator_pipeline = pipeline(
        "text-generation", # Task identifier for Hugging Face pipelines
        model=model,
        tokenizer=tokenizer,
        # device=0 # Uncomment this to attempt using GPU (if available and PyTorch with CUDA is installed)
    )
    print("Text generation pipeline created successfully.")

except Exception as e:
    # If any error occurs during loading (e.g., model not found, network issues, resource limits),
    # print an error message and set the global variables to None.
    # This allows the application to start but indicates that LLM generation will not be functional.
    print(f"Error loading model or tokenizer for {MODEL_NAME}: {e}")
    print("LLM generation will not be available. Check model name, internet connection, and resources.")
    tokenizer = None
    model = None
    generator_pipeline = None

# --- Core Function ---
def generate_response_from_context(query: str, context_chunks: list[str], max_context_length: int = 1500, max_new_tokens: int = 150) -> str:
    """
    Generates a response using the loaded language model based on a user query and retrieved context chunks.

    Args:
        query (str): The user's question or input.
        context_chunks (list[str]): A list of text snippets retrieved from the vector database,
                                     deemed relevant to the query.
        max_context_length (int): The maximum number of characters to use for the combined context
                                  in the prompt. Longer contexts will be truncated.
        max_new_tokens (int): The maximum number of new tokens the LLM is allowed to generate
                              as part of its response.

    Returns:
        str: The generated response from the language model. Returns an error message if generation fails
             or if the LLM pipeline is not available.
    """
    # Check if the generation pipeline was successfully initialized.
    if not generator_pipeline:
        return "Error: Text generation pipeline is not available. Model might have failed to load."

    # Construct the prompt for the LLM.
    if not context_chunks:
        # If no context chunks are provided (e.g., RAG retrieval found nothing relevant),
        # create a simpler prompt with just the query.
        # Alternatively, one might return a message indicating no context was found.
        prompt = f"Question: {query}\nAnswer:"
    else:
        # Combine all retrieved context chunks into a single string.
        combined_context = "\n".join(context_chunks)

        # Truncate the combined context if it exceeds the specified maximum length.
        # This is important to prevent overly long prompts that might exceed model limits or consume too many resources.
        if len(combined_context) > max_context_length:
            print(f"Warning: Combined context length ({len(combined_context)}) exceeds max_context_length ({max_context_length}). Truncating.")
            combined_context = combined_context[:max_context_length]

        # Construct the prompt incorporating the (potentially truncated) context and the query.
        # This format guides the LLM to answer the question based on the provided context.
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"

    print(f"\n--- Prompt for LLM (length: {len(prompt)}): ---\n{prompt}\n-------------------------")

    try:
        # Use the generator pipeline to produce text.
        # The pipeline handles tokenization of the prompt and decoding of the generated tokens.
        generated_outputs = generator_pipeline(
            prompt,
            max_length=len(tokenizer.encode(prompt)) + max_new_tokens, # Control total length of prompt + generated text
            num_return_sequences=1, # We only need one generated response.
            pad_token_id=tokenizer.eos_token_id # Use EOS token for padding if needed during generation.
        )

        # Extract the generated text from the pipeline's output.
        # The output is a list of dictionaries, each containing 'generated_text'.
        generated_text = generated_outputs[0]['generated_text']

        # Post-process the generated text to extract only the answer part.
        # LLMs sometimes repeat the input prompt in their output. This attempts to remove it.
        # A common strategy is to split the output by the original prompt and take the last part.
        answer = generated_text.split(prompt)[-1].strip()

        # Further refinement: If the above split didn't work well (e.g., prompt not exactly repeated,
        # or answer is empty), try to find "Answer:" marker if present in the prompt structure and take text after it.
        if prompt.strip() == answer.strip() or not answer: # Check if answer is just the prompt or empty
            answer_marker = "Answer:"
            marker_index = generated_text.rfind(answer_marker) # Find the last occurrence of "Answer:"
            if marker_index != -1:
                answer = generated_text[marker_index + len(answer_marker):].strip()
            else:
                 # If "Answer:" marker is not found, and the simple split by prompt failed,
                 # take the part of generated_text that is after the original prompt length.
                 # This is a fallback and might not always be perfect.
                 answer = generated_text[len(prompt):].strip()


        print(f"\n--- Raw LLM Output: ---\n{generated_text}\n-------------------------")
        print(f"\n--- Extracted Answer: ---\n{answer}\n-------------------------")

        # If the extracted answer is still empty, return a specific message.
        if not answer:
            return "The model generated an empty response after processing the context."

        return answer

    except Exception as e:
        # Catch any errors during the generation process.
        print(f"Error during text generation: {e}")
        return f"Error generating response: {e}"

# --- Example Usage (for direct testing of this module) ---
if __name__ == '__main__':
    # This block runs only when the script is executed directly (e.g., `python llm_generator.py`).
    # It's useful for testing the module's functionality independently.
    # Note: This requires the model to be successfully downloaded and loaded when the module is imported.

    if generator_pipeline: # Check if the pipeline was initialized correctly
        print("\nTesting LLM generation function...")

        sample_query = "What is solar power?"
        sample_context = [
            "Solar power is the conversion of energy from sunlight into electricity, either directly using photovoltaics (PV), indirectly using concentrated solar power, or a combination.",
            "Renewable energy sources include solar, wind, and geothermal power.",
            "The Earth receives an enormous amount of solar radiation."
        ]

        # Test with context
        generated_answer_with_context = generate_response_from_context(sample_query, sample_context)
        print(f"\nQuery: {sample_query}")
        print(f"Generated Answer (with context): {generated_answer_with_context}")

        print("\nTesting LLM generation with no context...")
        sample_query_no_context = "What is the capital of France?"
        # Test without context
        generated_answer_no_context = generate_response_from_context(sample_query_no_context, [])
        print(f"\nQuery: {sample_query_no_context}")
        print(f"Generated Answer (no context): {generated_answer_no_context}")

    else:
        # If the pipeline is not available, print a message indicating tests cannot be run.
        print("LLM Generator: Cannot run tests as the generation pipeline is not available.")
