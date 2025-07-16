import cv2
import json
import base64
import requests
import os
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import queue


class FoodRecognizerGUI:
    def __init__(self, config_file='food.json'):
        """Initialize the food recognizer with modern GUI."""
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

        # GUI setup
        self.root = tk.Tk()
        self.root.title("Food Type & Weight Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        # Variables
        self.is_analyzing = False
        self.frame_queue = queue.Queue()

        # Create GUI elements
        self.create_widgets()

        # Start camera feed
        self.update_camera()

        print("Food Recognition System with Modern UI initialized!")

    def configure_styles(self):
        """Configure modern styling for the GUI."""
        self.style.configure('Title.TLabel',
                             font=('Arial', 16, 'bold'),
                             background='#2c3e50',
                             foreground='#ecf0f1')

        self.style.configure('Header.TLabel',
                             font=('Arial', 12, 'bold'),
                             background='#34495e',
                             foreground='#ecf0f1',
                             padding=10)

        self.style.configure('Modern.TButton',
                             font=('Arial', 12, 'bold'),
                             padding=10)

        self.style.configure('Success.TButton',
                             background='#27ae60',
                             foreground='white')

        self.style.configure('Warning.TButton',
                             background='#e74c3c',
                             foreground='white')

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {config_file}")

    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main title
        title_label = ttk.Label(self.root, text="🍽️ Food Weight Recognition System",
                                style='Title.TLabel')
        title_label.pack(pady=10)

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left side - Camera feed
        left_frame = ttk.LabelFrame(main_frame, text="📹 Live Camera Feed",
                                    style='Header.TLabel')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Camera display
        self.camera_label = ttk.Label(left_frame, text="Loading camera...",
                                      background='#34495e', foreground='#ecf0f1')
        self.camera_label.pack(pady=20, padx=20)

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)

        self.analyze_button = ttk.Button(button_frame, text="📊 Analyze Food",
                                         command=self.analyze_food_async,
                                         style='Modern.TButton')
        self.analyze_button.pack(side='left', padx=5)

        self.save_button = ttk.Button(button_frame, text="💾 Save Image",
                                      command=self.save_current_frame,
                                      style='Modern.TButton')
        self.save_button.pack(side='left', padx=5)

        # Status label
        self.status_label = ttk.Label(left_frame, text="Ready to analyze food",
                                      background='#2c3e50', foreground='#95a5a6')
        self.status_label.pack(pady=5)

        # Right side - Results
        right_frame = ttk.LabelFrame(main_frame, text="📋 Analysis Results",
                                     style='Header.TLabel')
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Results display
        self.results_text = scrolledtext.ScrolledText(right_frame,
                                                      width=40, height=20,
                                                      font=('Consolas', 11),
                                                      background='#ecf0f1',
                                                      foreground='#2c3e50')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Clear results button
        clear_button = ttk.Button(right_frame, text="🗑️ Clear Results",
                                  command=self.clear_results,
                                  style='Warning.TButton')
        clear_button.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=5)

        # Initial message
        self.display_welcome_message()

    def display_welcome_message(self):
        """Display welcome message in results area."""
        welcome_text = """
🍽️ FOOD WEIGHT RECOGNITION SYSTEM
═══════════════════════════════════

Welcome! This system will help you identify food types and estimate their weight in grams.

HOW TO USE:
1. Position your food in front of the camera
2. Click "Analyze Food" to get weight estimation
3. Results will show food type and weight in grams

TIPS FOR BETTER ACCURACY:
• Use a standard plate or utensil for reference
• Ensure good lighting
• Keep food clearly visible
• Avoid shadows and glare

Ready to analyze your food!
        """
        self.results_text.insert(tk.END, welcome_text)

    def update_camera(self):
        """Update camera feed continuously."""
        ret, frame = self.cap.read()
        if ret:
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for display
            height, width = frame_rgb.shape[:2]
            new_width = 480
            new_height = int((new_width / width) * height)
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)

            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

            # Store current frame
            self.current_frame = frame

        # Schedule next update
        self.root.after(30, self.update_camera)

    def analyze_food_async(self):
        """Start food analysis in a separate thread."""
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.analyze_button.configure(state='disabled')
        self.status_label.configure(text="Analyzing food... Please wait")
        self.progress.start()

        # Start analysis in background thread
        thread = threading.Thread(target=self.analyze_food_thread)
        thread.daemon = True
        thread.start()

    def analyze_food_thread(self):
        """Analyze food in background thread."""
        try:
            # Encode current frame
            image_base64 = self.encode_image(self.current_frame)

            # Analyze with Gemini
            analysis = self.analyze_food_with_gemini(image_base64)

            # Update GUI in main thread
            self.root.after(0, self.display_analysis_results, analysis)

        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.root.after(0, self.display_analysis_results, error_msg)

    def encode_image(self, image):
        """Encode image to base64 for API submission."""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def analyze_food_with_gemini(self, image_base64):
        """Send image to Gemini API for food recognition."""
        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": """Analyze this image and identify food type and weight in GRAMS. Focus on:

FOOD TYPE:
- Identify each specific food item (e.g., "apple", "grilled chicken breast", "white rice")
- Be specific about preparation method if visible

WEIGHT ESTIMATION IN GRAMS:
- Use visual references (plates ~25cm diameter, forks ~18cm, hands for scale)
- Consider camera angle and perspective
- Provide weight estimates in GRAMS for each food item
- Use these reference weights for common foods:
  * Apple (medium): ~180g
  * Chicken breast: ~150-300g  
  * Rice (cooked, 1 cup): ~200g
  * Bread slice: ~25-30g
  * Banana: ~120g
  * Egg: ~50g

ANALYSIS APPROACH:
- Compare food size to visible reference objects
- Account for camera perspective distortion
- Consider typical serving sizes and densities
- Estimate total weight if multiple items

FORMAT YOUR RESPONSE EXACTLY AS:
FOOD TYPE: [specific food name]
WEIGHT: [estimated weight in grams]g
CONFIDENCE: [High/Medium/Low]

If multiple food items, list each separately.
If no food visible, respond: "NO FOOD DETECTED"

Focus ONLY on food type and weight in grams - be as accurate as possible."""
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
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def display_analysis_results(self, analysis):
        """Display analysis results in the GUI."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format results
        result_text = f"""
{'=' * 50}
🕐 ANALYSIS COMPLETED - {timestamp}
{'=' * 50}

{analysis}

{'=' * 50}

"""

        # Insert at the beginning of results
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)

        # Reset UI state
        self.is_analyzing = False
        self.analyze_button.configure(state='normal')
        self.status_label.configure(text="Analysis complete! Ready for next analysis")
        self.progress.stop()

        # Auto-save results
        self.save_analysis_to_file(analysis)

    def save_current_frame(self):
        """Save current camera frame."""
        if hasattr(self, 'current_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_image_{timestamp}.jpg"

            os.makedirs("captured_images", exist_ok=True)
            filepath = os.path.join("captured_images", filename)

            cv2.imwrite(filepath, self.current_frame)

            self.status_label.configure(text=f"Image saved: {filename}")
            messagebox.showinfo("Success", f"Image saved as:\n{filepath}")

    def save_analysis_to_file(self, analysis):
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"food_analysis_{timestamp}.txt"

        os.makedirs("analysis_results", exist_ok=True)
        filepath = os.path.join("analysis_results", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Food Weight Analysis - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis)

        print(f"Analysis saved: {filepath}")

    def clear_results(self):
        """Clear the results display."""
        self.results_text.delete(1.0, tk.END)
        self.display_welcome_message()

    def run(self):
        """Start the GUI application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"GUI error: {str(e)}")
            self.cleanup()

    def on_closing(self):
        """Handle window closing."""
        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the food recognition system."""
    try:
        app = FoodRecognizerGUI()
        app.run()
    except Exception as e:
        print(f"Error initializing food recognition system: {str(e)}")
        print("\nMake sure you have:")
        print("1. A 'food.json' file with your Gemini API key")
        print("2. A working webcam connected")
        print("3. Required packages: pip install opencv-python pillow requests")


if __name__ == "__main__":
    main()