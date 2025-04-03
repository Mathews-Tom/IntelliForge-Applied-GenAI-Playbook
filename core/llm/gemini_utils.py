# core/llm/gemini_utils.py

import os
from enum import Enum

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.generative_models import (
    GenerativeModel,
)  # More specific import for type hinting

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


def get_gemini_model(model_type: GeminiModelType) -> GenerativeModel:
    """
    Initializes and returns the specified Gemini GenerativeModel instance.

    Args:
        model_type: A GeminiModelType enum member specifying the model to use.

    Returns:
        An instance of google.generativeai.GenerativeModel.

    Raises:
        ValueError: If the Gemini API hasn't been configured.
        Exception: If there's an error initializing the model.
    """
    if not _is_configured:
        # Automatically configure if not already done
        print("Gemini API not configured. Attempting configuration...")
        configure_gemini_api()

    try:
        model_identifier = model_type.value
        print(f"Initializing Gemini model: {model_identifier}")
        model = genai.GenerativeModel(
            model_identifier
        )  # [real_time_data: Create the GenAI client]
        return model
    except Exception as e:
        print(
            f"Error initializing Gemini model {model_type.name} ({model_type.value}): {e}"
        )
        raise


def generate_content(model_type: GeminiModelType, prompt: str) -> str | None:
    """
    Generates content using the specified Gemini model.

    Args:
        model_type: The GeminiModelType enum member to use.
        prompt: The text prompt to send to the model.

    Returns:
        The generated text content as a string, or None if an error occurred.
    """
    try:
        model = get_gemini_model(model_type)
        print(f"Sending prompt to {model_type.name}...")
        response = model.generate_content(
            prompt
        )  # [real_time_data: Generate a response]
        # Basic error handling for response structure - adjust as needed based on API behavior
        if response and hasattr(response, "text"):
            return response.text
        else:
            print(
                f"Warning: Received unexpected response format from {model_type.name}: {response}"
            )
            # Attempt to access parts if available, otherwise return None
            if response and response.parts:
                return "".join(
                    part.text for part in response.parts if hasattr(part, "text")
                )
            return None

    except Exception as e:
        print(f"Error during content generation with {model_type.name}: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # This block executes only when the script is run directly
    # Create a .env file in the same directory with:
    # GOOGLE_API_KEY=your_actual_api_key

    print("Running Gemini Utils Example...")
    try:
        # Configure (will load .env)
        configure_gemini_api()  # Explicit call here, but get_gemini_model would also trigger it

        # Select the desired model
        # <<< MANDATORY REQUIREMENT: Use Gemini 2.5 Pro >>>
        selected_model = GeminiModelType.GEMINI_2_5_PRO  # [user query requirement]

        # Define a prompt
        test_prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in simple terms."

        # Generate content
        generated_text = generate_content(selected_model, test_prompt)

        if generated_text:
            print("\n--- Generated Content ---")
            print(generated_text)
            print("-------------------------\n")
        else:
            print("\nFailed to generate content.")

        # Example using another model
        # print("\nTrying with Gemini 1.5 Flash...")
        # flash_model = GeminiModelType.GEMINI_1_5_FLASH
        # flash_text = generate_content(flash_model, "What is the speed of light?")
        # if flash_text:
        #     print(flash_text)
        # else:
        #     print("Failed to generate content with Flash model.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
