import cv2
import json
import base64
import requests
import os
from datetime import datetime
import time


class FoodRecognizer:
    def __init__(self, config_file='food.json'):
        """Initialize the food recognizer with configuration from JSON file."""
        self.config = self.load_config(config_file)
        self.gemini_api_key = self.config.get('gemini_api_key')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found in food.json")

        # Gemini API endpoint
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")

        print("Food Recognition System initialized successfully!")
        print("Press 'SPACE' to capture and analyze food")
        print("Press 'q' to quit")

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {config_file}")

    def capture_image(self):
        """Capture an image from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image")
            return None
        return frame

    def encode_image(self, image):
        """Encode image to base64 for API submission."""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def analyze_food(self, image_base64):
        """Send image to Gemini API for food recognition."""
        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Analyze this image and identify the food items present. Please provide:\n1. The main food items you can identify\n2. Approximate portion sizes if possible\n3. Any nutritional insights or interesting facts about the foods\n4. If no food is visible, please say so clearly.\n\nBe specific and detailed in your analysis."
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }]
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "No analysis available from Gemini API"

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Gemini API: {str(e)}"
        except KeyError as e:
            return f"Unexpected response format from Gemini API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def save_image(self, image, analysis_text):
        """Save the captured image with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"food_analysis_{timestamp}.jpg"

        # Create directory if it doesn't exist
        os.makedirs("captured_images", exist_ok=True)
        filepath = os.path.join("captured_images", filename)

        cv2.imwrite(filepath, image)

        # Also save the analysis text
        text_filename = f"food_analysis_{timestamp}.txt"
        text_filepath = os.path.join("captured_images", text_filename)
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Food Analysis - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis_text)

        print(f"Image saved as: {filepath}")
        print(f"Analysis saved as: {text_filepath}")

    def run(self):
        """Main loop for the food recognition system."""
        print("\nStarting webcam feed...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Display the frame
            cv2.imshow('Food Recognition - Press SPACE to analyze, Q to quit', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space key pressed
                print("\nCapturing image...")

                # Capture and encode image
                image_base64 = self.encode_image(frame)

                print("Analyzing food with Gemini AI...")
                print("This may take a few seconds...")

                # Analyze with Gemini
                analysis = self.analyze_food(image_base64)

                # Display results
                print("\n" + "=" * 60)
                print("FOOD ANALYSIS RESULTS")
                print("=" * 60)
                print(analysis)
                print("=" * 60)

                # Save image and analysis
                self.save_image(frame, analysis)
                print("\nPress SPACE to analyze another image, or Q to quit")

            elif key == ord('q'):  # Q key pressed
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("Food recognition system closed.")


def main():
    """Main function to run the food recognition system."""
    try:
        recognizer = FoodRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error initializing food recognition system: {str(e)}")
        print("\nMake sure you have:")
        print("1. A 'food.json' file with your Gemini API key")
        print("2. A working webcam connected")
        print("3. Required Python packages installed: cv2, requests")


if __name__ == "__main__":
    main()