# core/llm/gemini_utils.py

import asyncio  # Added for async operations
import functools  # Added for caching
import os
from enum import Enum
from typing import AsyncIterator, Union  # Added for streaming type hints

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from tenacity import (
    RetryError,
    retry,  # Added for retries
    stop_after_attempt,
    wait_exponential,
)

# --- Gemini Model Selection Enum ---


class GeminiModelType(Enum):
    """
    Enumeration of available Google Gemini models.
    The values should correspond to the model identifiers used by the Google AI API.
    Verify these against the official Google AI documentation.
    """

    GEMINI_2_5_PRO = "models/gemini-2.5-pro-exp-03-25"  # Gemini 2.5 Pro Experimental
    GEMINI_2_0_FLASH = "models/gemini-2.0-flash"  # Gemini 2.0 Flash
    GEMINI_1_5_PRO = "models/gemini-1.5-pro"  # Gemini 1.5 Pro
    GEMINI_1_5_FLASH = "models/gemini-1.5-flash"  # Gemini 1.5 Flash
    # Add other Gemini model identifiers as needed (e.g., older versions owr fine-tuned models)

    def __str__(self):
        """Returns the string value of the enum member."""
        return self.value


# --- Core Gemini Functions ---

_is_configured = False


def configure_gemini_api() -> None:
    """
    Configures the Google Generative AI client using the API key
    from environment variables. Should be called once.
    """
    global _is_configured
    if _is_configured:
        return

    # Loads environment variables from a .env file if present.
    load_dotenv()  # Loads variables from .env file into environment [1][3]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API Key not found. Set the GOOGLE_API_KEY environment "
            "variable, preferably in a .env file."
        )
    try:
        genai.configure(api_key=api_key)
        _is_configured = True
        print("Gemini AI client configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini AI client: {e}")
        raise


@functools.lru_cache()  # Cache model instances
def get_gemini_model(model_type: GeminiModelType) -> GenerativeModel:
    """
    Initializes (if not cached) and returns the specified Gemini GenerativeModel instance.

    Args:
        model_type: A GeminiModelType enum member specifying the model to use.

    Returns:
        A cached instance of google.generativeai.GenerativeModel.

    Raises:
        ValueError: If the Gemini API hasn't been configured.
        Exception: If there's an error initializing the model.
    """
    if not _is_configured:
        # Automatically configure if not already done
        # This configuration check still needs to run before the first call
        print("Gemini API not configured. Attempting configuration...")
        configure_gemini_api()

    try:
        model_identifier = model_type.value
        # Removed print statement as it's less relevant with caching
        model = genai.GenerativeModel(
            model_identifier
        )  # [real_time_data: Create the GenAI client]
        print(
            f"Gemini model '{model_identifier}' initialized (or retrieved from cache)."
        )
        return model
    except Exception as e:
        print(
            f"Error initializing Gemini model {model_type.name} ({model_type.value}): {e}"
        )
        raise


@retry(
    wait=wait_exponential(
        multiplier=1, min=2, max=60
    ),  # Exponential backoff: 2s, 4s, 8s... up to 60s
    stop=stop_after_attempt(3),  # Stop after 3 attempts (1 initial + 2 retries)
    reraise=True,  # Reraise the exception if all retries fail
)
async def generate_content(
    model_type: GeminiModelType, prompt: str, stream: bool = False
) -> Union[str, AsyncIterator[str], None]:
    """
    Generates content asynchronously using the specified Gemini model,
    with automatic retries on failure. Can optionally stream the response.

    Args:
        model_type: The GeminiModelType enum member to use.
        prompt: The text prompt to send to the model.
        stream: If True, returns an async iterator yielding text chunks.
                If False (default), returns the complete text as a string.

    Returns:
        - If stream=False: The generated text content as a string, or None if retries fail or response is invalid.
        - If stream=True: An async iterator yielding text chunks, or None if the initial call fails.

    Raises:
        RetryError: If all retry attempts fail for the initial API call.
        Exception: Other exceptions during model interaction.
    """
    try:
        # get_gemini_model is synchronous and cached, so it's fine to call directly
        model = get_gemini_model(model_type)
        print(
            f"Attempting to send prompt asynchronously to {model_type.name} (stream={stream})..."
        )

        # Assuming the library provides an async method, e.g., generate_content_async
        # Replace 'generate_content_async' if the actual method name differs
        response_iterator = await model.generate_content_async(
            prompt, stream=stream
        )  # [real_time_data: Generate a response]

        if stream:
            # Define an inner async generator to yield text chunks
            async def _stream_generator(iterator):
                try:
                    async for chunk in iterator:
                        if hasattr(chunk, "text"):
                            yield chunk.text
                        elif chunk.parts:  # Handle potential alternative structure
                            yield "".join(
                                p.text for p in chunk.parts if hasattr(p, "text")
                            )
                except Exception as stream_error:
                    print(
                        f"Error during streaming from {model_type.name}: {stream_error}"
                    )
                    # Decide how to handle stream errors: yield None, raise, or just stop
                    yield f"\n[Error during stream: {stream_error}]"  # Example: yield error message

            print(f"Successfully initiated stream from {model_type.name}.")
            return _stream_generator(response_iterator)
        else:
            # Handle non-streaming response (already awaited)
            response = response_iterator  # In non-streaming, the awaited result is the full response
            if response and hasattr(response, "text"):
                print(
                    f"Successfully received non-streamed response from {model_type.name}."
                )
                return response.text
            elif response and response.parts:
                print(
                    f"Successfully received non-streamed response (parts) from {model_type.name}."
                )
                return "".join(
                    part.text for part in response.parts if hasattr(part, "text")
                )
            else:
                print(
                    f"Warning: Received unexpected non-streamed response format from {model_type.name}: {response}"
                )
                return None

    except RetryError as e:
        # Logged by tenacity by default, but we can add specific logging
        print(f"API call to {model_type.name} failed after multiple retries: {e}")
        return None  # Return None after retries are exhausted
    except Exception as e:
        # Catch other potential exceptions not handled by retry (e.g., config errors)
        print(
            f"An unexpected error occurred during content generation with {model_type.name}: {e}"
        )
        return None  # Keep consistent return type on failure


# --- Example Usage ---
if __name__ == "__main__":
    # This block executes only when the script is run directly
    # Create a .env file in the same directory with:
    # GOOGLE_API_KEY=your_actual_api_key

    async def main():
        print("Running Gemini Utils Async Example...")
        try:
            # Configure (will load .env)
            # configure_gemini_api() is called implicitly by get_gemini_model if needed

            # Select the desired model
            # <<< MANDATORY REQUIREMENT: Use Gemini 2.5 Pro >>>
            selected_model = GeminiModelType.GEMINI_2_5_PRO  # [user query requirement]

            # Define a prompt
            test_prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in simple terms."

            # Generate content asynchronously
            print("\nGenerating content with Gemini 2.5 Pro...")
            generated_text = await generate_content(selected_model, test_prompt)

            if generated_text:
                print("\n--- Generated Content (2.5 Pro) ---")
                print(generated_text)
                print("-----------------------------------\n")
            else:
                print("\nFailed to generate content with 2.5 Pro.")

            # Example using another model (async)
            print("\nTrying with Gemini 1.5 Flash...")
            flash_model = GeminiModelType.GEMINI_1_5_FLASH
            flash_prompt = "What is the speed of light in vacuum?"
            flash_text = await generate_content(flash_model, flash_prompt)
            if flash_text:
                print("\n--- Generated Content (1.5 Flash) ---")
                print(flash_text)
                print("-------------------------------------\n")
            else:
                print("Failed to generate content with Flash model.")

        except ValueError as ve:
            print(f"Configuration Error: {ve}")
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}")

    # Run the async main function
    asyncio.run(main())

    # --- Streaming Example ---
    async def stream_example():
        print("\n--- Running Streaming Example ---")
        try:
            model = GeminiModelType.GEMINI_1_5_FLASH
            prompt = "Tell me a short story about a curious robot exploring a garden."
            print(f"Streaming response from {model.name}...")

            response_stream = await generate_content(model, prompt, stream=True)

            if response_stream:
                full_response = ""
                async for chunk in response_stream:
                    print(chunk, end="", flush=True)  # Print chunks as they arrive
                    full_response += chunk
                print("\n--- End of Stream ---")
                # print(f"\nFull streamed response:\n{full_response}") # Optional: print full assembled text
            else:
                print("Failed to initiate stream.")

        except Exception as ex:
            print(f"\nAn error occurred during the streaming example: {ex}")

    print("\n")
    asyncio.run(stream_example())
