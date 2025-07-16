import cv2
import json
import time
import google.generativeai as genai # Keep this as the primary import alias
import traceback

# --- Configuration ---
API_KEY_FILE = "food.json"
GEMINI_MODEL_NAME = 'models/gemini-2.5-flash'
PROMPT = "tell me what's the food in this image and estimate the quantity"
API_CALL_INTERVAL_SECONDS = 5 # How often to call the Gemini API
WEBCAM_INDEX = 0 # Usually 0 for the default webcam
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0, 255, 0) # Green BGR
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

# --- API Key Loading ---
def load_api_key(json_path: str = API_KEY_FILE) -> str | None:
    """Loads the Gemini API key from a JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if "gemini_api_key" in data:
                return data["gemini_api_key"]
            else:
                print(f"Error: No 'gemini_api_key' found in {json_path}.")
                return None
    except FileNotFoundError:
        print(f"Error: '{json_path}' not found. Please create it with your Gemini API key.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'. Check its format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading API key: {e}")
        return None

# --- Gemini API Configuration ---
def configure_gemini(api_key: str):
    """Configures the Google Generative AI library with the provided API key."""
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to configure Gemini API. Please check your API key and network connection: {e}")
        exit(1) # Exit if API cannot be configured

# --- Food Analysis Function ---
def analyze_food_with_gemini(image_bytes: bytes) -> str:
    """
    Sends image bytes to Gemini for food identification and quantity estimation.

    Args:
        image_bytes: The image data in bytes (e.g., from cv2.imencode).

    Returns:
        The text response from Gemini, or an error message.
    """
    try:
        # Revert to the dictionary structure for image data, which is universally supported
        content_parts = [
            {"type": "text", "text": PROMPT},
            {
                "type": "image_data",
                "mime_type": "image/jpeg", # This must accurately reflect the image encoding
                "data": image_bytes
            }
        ]

        # Initialize the GenerativeModel from the 'genai' alias
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Generate content with the multimodal input
        response = model.generate_content(content_parts)

        # Return the text content from the response
        return response.text

    except genai.types.BlockedPromptException as e:
        # This exception is raised if the content is flagged by safety filters
        print(f"Warning: Content blocked by safety filters. Feedback: {e.response.prompt_feedback}")
        return "Gemini API: Content blocked by safety filter."
    except Exception as e:
        # Catch any other API or network errors
        print(f"Detailed API error during analysis: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return f"Gemini API Error: {str(e)}"

# --- Main Application Logic ---
def main():
    # 1. Load API Key
    api_key = load_api_key()
    if not api_key:
        print("Exiting due to missing or invalid API key.")
        return

    # 2. Configure Gemini API
    configure_gemini(api_key)

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"FATAL ERROR: Cannot open webcam at index {WEBCAM_INDEX}. Please check if it's connected and not in use.")
        return

    # --- Live Loop Variables ---
    current_description = "Initializing Gemini analysis..."
    last_api_call_time = 0

    print("\n--- Starting Webcam Feed ---")
    print(f"Gemini API will be called every {API_CALL_INTERVAL_SECONDS} seconds.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting.")
            break

        current_time = time.time()

        # Call Gemini API periodically
        if current_time - last_api_call_time > API_CALL_INTERVAL_SECONDS:
            # Encode the OpenCV frame to JPEG bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                image_bytes = buffer.tobytes()
                # Debugging print: Check if bytes are actually generated
                if not image_bytes:
                    current_description = "Error: Encoded image is empty."
                    print("Warning: Encoded image bytes are empty!")
                else:
                    current_description = analyze_food_with_gemini(image_bytes)
            else:
                current_description = "Error: Failed to encode image to JPEG."
            last_api_call_time = current_time

        # Display the description on the frame
        if not isinstance(current_description, str):
            current_description = str(current_description)

        text_lines = current_description.split('\n')
        y_offset = 30
        line_height = 25

        for i, line in enumerate(text_lines):
            y_pos = y_offset + i * line_height
            cv2.putText(frame, line, (10, y_pos), FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Gemini Food Vision', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\nScript terminated.")

if __name__ == "__main__":
    main()