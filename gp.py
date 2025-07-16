import os
import cv2 # Import OpenCV for webcam access
from google.cloud import vision
from google.oauth2 import service_account
import time # For controlling API call frequency

# --- Configuration ---
# IMPORTANT: Place your Google Cloud service account JSON key file
# in the same directory as this script and specify its name here.
credentials_file_name = "food.json" # <--- RENAME THIS to your actual JSON key file name

# --- Vision API Client Initialization (moved outside function for efficiency) ---
# This client will be initialized once and reused for all API calls.
vision_client = None
credentials_path_global = None

def initialize_vision_client(credentials_path):
    """Initializes the Google Cloud Vision API client."""
    global vision_client, credentials_path_global
    if vision_client is None or credentials_path != credentials_path_global:
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            credentials_path_global = credentials_path
            print("Google Cloud Vision API client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Vision API client: {e}")
            print("Please ensure your credentials file is valid and accessible.")
            vision_client = None # Reset client on error
            raise # Re-raise to stop execution if client cannot be initialized

# --- Function to detect food in image bytes ---
def detect_food_in_bytes(image_bytes):
    """
    Detects labels in image bytes using Google Cloud Vision API and
    filters for food-related concepts.

    Args:
        image_bytes (bytes): The image content as bytes (e.g., from a webcam frame).

    Returns:
        list: A list of strings, where each string is a detected food-related label.
              Returns an empty list if no food is detected or an error occurs.
    """
    if vision_client == None:
        print("Vision API client not initialized. Cannot detect food.")
        return []

    try:
        image = vision.Image(content=image_bytes)

        # --- CORE VISION API USAGE ---
        # This is where the script leverages the Google Cloud Vision API.
        # It performs label detection, which identifies broad categories and objects
        # present in the image, including food items.
        response = vision_client.label_detection(image=image)
        labels = response.label_annotations

        food_labels = []
        # Expanded food_keywords for more specific detection
        food_keywords = [
            "food", "cuisine", "dish", "ingredient", "meal", "produce",
            "vegetable", "fruit", "meat", "dessert", "snack", "beverage",
            "cooking", "recipe", "fast food", "street food", "comfort food",
            "baked goods", "dairy", "seafood", "poultry", "grill", "soup",
            "salad", "pasta", "pizza", "bread", "cake", "cookie", "drink",
            "apple", "banana", "orange", "grape", "strawberry", "blueberry",
            "raspberry", "mango", "pineapple", "kiwi", "melon", "watermelon",
            "peach", "pear", "cherry", "lemon", "lime", "avocado", "tomato",
            "potato", "carrot", "broccoli", "spinach", "lettuce", "cucumber",
            "onion", "garlic", "pepper", "mushroom", "corn", "bean", "pea",
            "rice", "noodle", "chicken", "beef", "pork", "fish", "shrimp",
            "egg", "cheese", "milk", "yogurt", "butter", "chocolate", "candy",
            "sandwich", "burger", "sushi", "taco", "burrito", "curry", "stew",
            "pie", "donut", "muffin", "croissant", "juice", "coffee", "tea",
            "soda", "water"
        ]

        # print(f"Detected labels:") # Uncomment for verbose output
        for label in labels:
            # print(f"- {label.description} (Score: {label.score:.2f})") # Uncomment for verbose output
            # The script then processes these labels to identify food-specific items.
            # This filtering uses a predefined list of food-related keywords
            # and confidence scores to refine the Vision API's general labels
            # into relevant food categories.
            if any(keyword in label.description.lower() for keyword in food_keywords) or \
               (label.score > 0.7 and label.description.lower() in food_keywords):
                food_labels.append(label.description)
            elif label.score > 0.8 and ("food" in label.mid.lower() or "dish" in label.mid.lower() or "cuisine" in label.mid.lower()):
                 # Check for more general food-related entity IDs if available and high confidence
                 food_labels.append(label.description)


        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")

        return list(set(food_labels)) # Return unique food labels

    except Exception as e:
        print(f"An error occurred during Vision API call: {e}")
        print("Please ensure:")
        print("1. The Cloud Vision API is enabled for your Google Cloud project.")
        print("2. The service account associated with the key has the 'Cloud Vision API User' role.")
        return []

# --- Function to process live webcam feed ---
def process_webcam_feed(credentials_path):
    """
    Captures video from webcam, sends frames to Vision API for food detection,
    and displays results.
    """
    # Initialize Vision API client
    try:
        initialize_vision_client(credentials_path)
    except Exception:
        return # Exit if client cannot be initialized

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is typically the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected and not in use.")
        return

    print("\n--- Live Webcam Food Detection ---")
    print("Press 'q' to quit.")

    last_api_call_time = time.time()
    api_call_interval = 2.0 # Call API every 2 seconds to avoid hitting rate limits too quickly
    detected_food_items = []

    while True:
        ret, frame = cap.read() # Read a frame from the webcam

        if not ret:
            print("Failed to grab frame.")
            break

        current_time = time.time()

        # Only call Vision API periodically
        if current_time - last_api_call_time >= api_call_interval:
            # Convert frame to JPEG bytes (Vision API prefers JPEG/PNG)
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                image_bytes = buffer.tobytes()
                print(f"\nCalling Vision API for frame at {time.strftime('%H:%M:%S', time.localtime(current_time))}...")
                detected_food_items = detect_food_in_bytes(image_bytes)
                last_api_call_time = current_time
            else:
                print("Error: Could not encode frame to JPEG.")

        # Display detected food items on the frame
        y_offset = 30
        for item in detected_food_items:
            cv2.putText(frame, item, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 25

        # Display the frame
        cv2.imshow('Live Food Detector (Press "q" to quit)', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam feed stopped.")


# --- Example Usage ---
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct full path for the credentials file
    full_credentials_path = os.path.join(script_dir, credentials_file_name)

    # Check if credentials file exists
    if not os.path.exists(full_credentials_path):
        print(f"\n--- IMPORTANT ---")
        print(f"Error: Google Cloud credentials file not found!")
        print(f"Please place your service account JSON key file (e.g., '{credentials_file_name}')")
        print(f"in the same directory as this script: {script_dir}")
        print(f"You can download this file from Google Cloud Console > IAM & Admin > Service Accounts > Your Service Account > Keys > Add Key > Create new key (JSON).")
        print(f"-----------------\n")
    else:
        # Run the webcam processing function
        process_webcam_feed(full_credentials_path)

        # The previous file-based example is commented out below.
        # Uncomment and adjust if you still need to process a static image.
        # image_file_name = "meal.jpg" # Image to process, assumed to be in the same folder
        # full_image_path = os.path.join(script_dir, image_file_name)
        # if not os.path.exists(full_image_path):
        #     print(f"\n--- IMPORTANT ---")
        #     print(f"Error: Image file not found!")
        #     print(f"Please ensure '{image_file_name}' is in the same directory as this script: {script_dir}")
        #     print(f"-----------------\n")
        # else:
        #     print(f"Attempting to detect food in: {full_image_path}")
        #     print(f"Using credentials from: {full_credentials_path}")
        #     detected_food = detect_food_in_image(full_image_path, full_credentials_path) # This function is no longer defined for file-based input
        #
        #     if detected_food:
        #         print("\n--- Detected Food Items ---")
        #         for item in detected_food:
        #             print(f"- {item}")
        #     else:
        #         print("\nNo specific food items detected or an error occurred.")
        #         print("Consider refining the 'food_keywords' list or checking the image content.")

