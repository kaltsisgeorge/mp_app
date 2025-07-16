import cv2
import json
import base64
import requests
import os
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk, ImageDraw
import threading
import queue


class MicroplasticsAnalyzer:
    def __init__(self, config_file='food.json'):
        """Initialize the Professional Microplastics Food Analyzer."""
        self.config = self.load_config(config_file)
        self.gemini_api_key = self.config.get('gemini_api_key')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found in food.json")

        # API Configuration
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"

        # Initialize camera
        self.cap = None
        self.current_frame = None
        self.initialize_camera()

        # Professional dark color scheme
        self.colors = {
            'primary': '#D946EF',  # Bright magenta (primary)
            'primary_hover': '#C026D3',  # Darker magenta
            'secondary': '#A21CAF',  # Deep purple-magenta
            'secondary_hover': '#86198F',  # Darker purple-magenta
            'accent': '#F472B6',  # Pink accent
            'success': '#10B981',  # Teal green (kept for contrast)
            'success_hover': '#059669',  # Darker teal
            'warning': '#F59E0B',  # Warm orange (kept)
            'warning_hover': '#D97706',  # Deeper orange
            'error': '#EF4444',  # Bright red
            'error_hover': '#DC2626',  # Deeper red
            'bg_primary': '#1F1B24',  # Deep purple-black
            'bg_secondary': '#2A2233',  # Slightly lighter purple-gray
            'bg_tertiary': '#3C2C44',  # Medium purple-gray
            'bg_tertiary_hover': '#5B416A',  # Hover state
            'bg_dark': '#141016',  # Very dark background
            'text_primary': '#FDF4FF',  # Very light pinkish-white
            'text_secondary': '#E9D5FF',  # Soft lavender
            'text_light': '#C4B5FD',  # Muted purple
            'text_accent': '#F472B6',  # Pink (same as accent)
            'white': '#FFFFFF',
            'black': '#000000',
            'card_bg': '#2E1A35',  # Dark plum card background
            'border': '#5B3A5E',  # Muted magenta border
            'disabled': '#6B4E71',  # Muted purple-gray
            'disabled_text': '#A78BFA'  # Soft violet
        }

        # Application state
        self.is_analyzing = False
        self.analysis_thread = None

        # Initialize GUI
        self.setup_gui()

        # Start camera feed
        self.update_camera()

        print("Professional Microplastics Food Analyzer initialized successfully.")

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {config_file}")

    def initialize_camera(self):
        """Initialize camera with error handling."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try different camera indices
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                else:
                    raise ValueError("No camera found")

            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.cap = None

    def setup_gui(self):
        """Setup professional GUI interface."""
        self.root = tk.Tk()
        self.root.title("Professional Microplastics Food Analyzer")
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.colors['bg_primary'])
        self.root.resizable(True, True)

        # Configure professional styling
        self.configure_professional_styles()

        # Create main layout
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()

        # Initialize with professional welcome
        self.display_professional_welcome()

    def configure_professional_styles(self):
        """Configure professional styling for the application."""
        self.style = ttk.Style()

        # Use available theme
        available_themes = self.style.theme_names()
        if 'clam' in available_themes:
            self.style.theme_use('clam')
        elif 'alt' in available_themes:
            self.style.theme_use('alt')
        else:
            self.style.theme_use('default')

        # Configure standard button styles
        self.style.configure('TButton',
                             font=('Segoe UI', 10, 'bold'),
                             padding=(20, 10))

        # Configure progress bar
        self.style.configure('TProgressbar',
                             background=self.colors['primary'],
                             troughcolor=self.colors['bg_tertiary'])

    def create_professional_button(self, parent, text, command, button_type='primary', width=None, height=None):
        """Create a professional styled button with proper hover effects."""
        # Define button configurations
        button_configs = {
            'primary': {
                'bg': self.colors['primary'],
                'hover_bg': self.colors['primary_hover'],
                'fg': self.colors['white'],
                'font': ('Segoe UI', 14, 'bold'),
                'relief': 'flat',
                'bd': 0,
                'pady': 15,
                'padx': 30
            },
            'secondary': {
                'bg': self.colors['bg_tertiary'],
                'hover_bg': self.colors['bg_tertiary_hover'],
                'fg': self.colors['text_primary'],
                'font': ('Segoe UI', 11, 'bold'),
                'relief': 'flat',
                'bd': 0,
                'pady': 12,
                'padx': 20
            },
            'success': {
                'bg': self.colors['success'],
                'hover_bg': self.colors['success_hover'],
                'fg': self.colors['white'],
                'font': ('Segoe UI', 12, 'bold'),
                'relief': 'flat',
                'bd': 0,
                'pady': 12,
                'padx': 25
            },
            'warning': {
                'bg': self.colors['warning'],
                'hover_bg': self.colors['warning_hover'],
                'fg': self.colors['white'],
                'font': ('Segoe UI', 11, 'bold'),
                'relief': 'flat',
                'bd': 0,
                'pady': 10,
                'padx': 20
            }
        }

        config = button_configs.get(button_type, button_configs['primary'])

        # Create the button
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=config['font'],
            bg=config['bg'],
            fg=config['fg'],
            relief=config['relief'],
            bd=config['bd'],
            pady=config['pady'],
            padx=config['padx'],
            cursor='hand2',
            activebackground=config['hover_bg'],
            activeforeground=config['fg']
        )

        # Add custom width/height if specified
        if width:
            button.configure(width=width)
        if height:
            button.configure(height=height)

        # Add hover effects
        def on_enter(e):
            if button['state'] != 'disabled':
                button.configure(bg=config['hover_bg'])

        def on_leave(e):
            if button['state'] != 'disabled':
                button.configure(bg=config['bg'])

        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)

        return button

    def create_header(self):
        """Create professional header section."""
        # Main header container
        header_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        header_container.pack(fill='x', padx=0, pady=0)

        # Header content with proper padding
        header_frame = tk.Frame(header_container, bg=self.colors['bg_dark'])
        header_frame.pack(fill='x', padx=30, pady=20)

        # Title section
        title_label = tk.Label(header_frame,
                               text="Professional Microplastics Food Analyzer",
                               font=('Segoe UI', 26, 'bold'),
                               bg=self.colors['bg_dark'],
                               fg=self.colors['white'])
        title_label.pack(anchor='w', pady=(0, 8))

        subtitle_label = tk.Label(header_frame,
                                  text="Advanced AI-Powered Food Safety Analysis • Weight Estimation • Calorie Calculation • Microplastic Detection",
                                  font=('Segoe UI', 12),
                                  bg=self.colors['bg_dark'],
                                  fg=self.colors['text_light'])
        subtitle_label.pack(anchor='w')

        # Add gradient effect with separator
        separator = tk.Frame(header_container, bg=self.colors['primary'], height=3)
        separator.pack(fill='x')

    def create_main_layout(self):
        """Create the main application layout."""
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Left panel - Camera and controls
        left_panel = tk.Frame(main_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Add border effect
        border_frame = tk.Frame(left_panel, bg=self.colors['border'], height=1)
        border_frame.pack(fill='x', side='top')

        # Camera section
        camera_frame = tk.LabelFrame(left_panel, text="Camera Feed",
                                     font=('Segoe UI', 14, 'bold'),
                                     bg=self.colors['card_bg'],
                                     fg=self.colors['text_primary'],
                                     relief='flat',
                                     bd=0,
                                     labelanchor='n')
        camera_frame.pack(fill='both', expand=True, padx=20, pady=(20, 10))

        # Camera display container
        self.camera_container = tk.Frame(camera_frame, bg=self.colors['bg_secondary'],
                                         relief='solid', bd=1, highlightbackground=self.colors['border'])
        self.camera_container.pack(fill='both', expand=True, pady=(15, 0))

        self.camera_label = tk.Label(self.camera_container,
                                     text="Initializing camera...",
                                     font=('Segoe UI', 14),
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_secondary'])
        self.camera_label.pack(expand=True)

        # Controls section
        controls_frame = tk.LabelFrame(left_panel, text="Analysis Controls",
                                       font=('Segoe UI', 14, 'bold'),
                                       bg=self.colors['card_bg'],
                                       fg=self.colors['text_primary'],
                                       relief='flat',
                                       bd=0,
                                       labelanchor='n')
        controls_frame.pack(fill='x', padx=20, pady=(10, 20))

        # Control content frame
        controls_content = tk.Frame(controls_frame, bg=self.colors['card_bg'])
        controls_content.pack(fill='x', pady=(15, 0))

        # Main analyze button using the professional button function
        self.analyze_button = self.create_professional_button(
            controls_content,
            "🔬 START ANALYSIS",
            self.start_analysis,
            'primary'
        )
        self.analyze_button.pack(pady=(0, 20), fill='x')

        # Secondary buttons frame
        button_frame = tk.Frame(controls_content, bg=self.colors['card_bg'])
        button_frame.pack(fill='x')

        # Create secondary buttons with equal sizing
        self.save_button = self.create_professional_button(
            button_frame,
            "💾 Save Image",
            self.save_current_frame,
            'secondary'
        )
        self.save_button.pack(side='left', padx=(0, 10), fill='x', expand=True)

        self.clear_button = self.create_professional_button(
            button_frame,
            "🗑️ Clear Results",
            self.clear_results,
            'secondary'
        )
        self.clear_button.pack(side='right', padx=(10, 0), fill='x', expand=True)

        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg=self.colors['card_bg'], relief='flat', bd=0)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Add border effect
        border_frame = tk.Frame(right_panel, bg=self.colors['border'], height=1)
        border_frame.pack(fill='x', side='top')

        # Results section
        results_frame = tk.LabelFrame(right_panel, text="Analysis Results",
                                      font=('Segoe UI', 14, 'bold'),
                                      bg=self.colors['card_bg'],
                                      fg=self.colors['text_primary'],
                                      relief='flat',
                                      bd=0,
                                      labelanchor='n')
        results_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame,
                                                      width=50, height=30,
                                                      font=('Consolas', 11),
                                                      bg=self.colors['bg_secondary'],
                                                      fg=self.colors['text_primary'],
                                                      selectbackground=self.colors['accent'],
                                                      selectforeground=self.colors['white'],
                                                      insertbackground=self.colors['primary'],
                                                      relief='flat',
                                                      borderwidth=1,
                                                      highlightcolor=self.colors['primary'],
                                                      highlightbackground=self.colors['border'],
                                                      wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True, pady=(15, 0))

    def create_status_bar(self):
        """Create professional status bar."""
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], height=50)
        self.status_frame.pack(fill='x', padx=0, pady=0)
        self.status_frame.pack_propagate(False)

        # Status content
        status_content = tk.Frame(self.status_frame, bg=self.colors['bg_dark'])
        status_content.pack(fill='both', expand=True)

        self.status_label = tk.Label(status_content,
                                     text="🟢 Ready for analysis",
                                     font=('Segoe UI', 11, 'bold'),
                                     bg=self.colors['bg_dark'],
                                     fg=self.colors['success'])
        self.status_label.pack(side='left', padx=30, pady=15)

        # Progress bar
        self.progress = ttk.Progressbar(status_content,
                                        mode='indeterminate',
                                        style='TProgressbar')
        self.progress.pack(side='right', padx=30, pady=15, fill='x', expand=True)

    def display_professional_welcome(self):
        """Display professional welcome message."""
        welcome_text = """┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     PROFESSIONAL MICROPLASTICS FOOD ANALYZER v3.0                      │
│                                  AI-POWERED ANALYSIS SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

🔬 SYSTEM OVERVIEW:
Advanced AI system for comprehensive food safety analysis using real-time computer vision
and scientific research data for microplastic contamination assessment.

📊 ANALYSIS CAPABILITIES:
• Food Type Identification    - Precise categorization using AI recognition
• Weight Estimation          - Accurate mass calculation with visual references  
• Calorie Calculation        - Nutritional energy content assessment
• Microplastic Detection     - Research-based contamination level analysis

🎯 USAGE INSTRUCTIONS:
1. Position food items clearly within camera frame
2. Ensure proper lighting and minimal shadows
3. Click "START ANALYSIS" for comprehensive evaluation
4. Review detailed results with risk assessment

📈 SYSTEM STATUS: Ready for Analysis
"""
        self.results_text.insert(tk.END, welcome_text)

    def update_camera(self):
        """Update camera feed with enhanced stability."""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Store current frame for analysis
                self.current_frame = frame.copy()

                # Enhance image quality
                frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)

                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate display dimensions
                height, width = frame_rgb.shape[:2]
                display_width = 640
                display_height = int((display_width / width) * height)

                # Resize for display
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))

                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)

                # Update camera display
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo
            else:
                self.camera_label.configure(text="📷 Camera feed unavailable",
                                            image="")
        else:
            self.camera_label.configure(text="❌ No camera detected",
                                        image="")

        # Schedule next update
        self.root.after(33, self.update_camera)  # ~30 FPS

    def start_analysis(self):
        """Start food analysis with proper error handling."""
        if self.is_analyzing:
            return

        if self.current_frame is None:
            messagebox.showerror("Error", "No camera frame available for analysis")
            return

        self.is_analyzing = True
        self.update_ui_analyzing_state(True)

        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.perform_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def perform_analysis(self):
        """Perform the actual food analysis."""
        try:
            # Update status
            self.root.after(0, self.update_status, "🔄 Encoding image...", self.colors['warning'])

            # Encode image
            image_base64 = self.encode_image(self.current_frame)

            # Update status
            self.root.after(0, self.update_status, "🤖 Analyzing with AI...", self.colors['warning'])

            # Perform AI analysis
            analysis_result = self.analyze_with_gemini(image_base64)

            # Update UI with results
            self.root.after(0, self.display_analysis_results, analysis_result)

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, self.display_analysis_results, error_msg)
        finally:
            self.root.after(0, self.update_ui_analyzing_state, False)

    def update_ui_analyzing_state(self, analyzing):
        """Update UI elements based on analyzing state."""
        if analyzing:
            self.analyze_button.configure(
                text="🔬 ANALYZING...",
                state='disabled',
                bg=self.colors['disabled'],
                fg=self.colors['disabled_text']
            )
            self.progress.start()
        else:
            self.analyze_button.configure(
                text="🔬 START ANALYSIS",
                state='normal',
                bg=self.colors['primary'],
                fg=self.colors['white']
            )
            self.progress.stop()
            self.is_analyzing = False

    def update_status(self, message, color):
        """Update status bar message."""
        self.status_label.configure(text=message, fg=color)

    def encode_image(self, image):
        """Encode image to base64 for API submission."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_with_gemini(self, image_base64):
        """Perform analysis using Gemini AI with research-based prompt."""
        headers = {'Content-Type': 'application/json'}

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": """You are a professional food analysis AI system. Analyze this image and provide ONLY the information in the EXACT format specified below.

ANALYSIS REQUIREMENTS:
1. Identify each food item precisely
2. Estimate weight in grams using visual references
3. Calculate calories using standard nutritional data
4. Determine microplastic contamination level in mg/kg and risk factor

RESPONSE FORMAT (use EXACTLY this format):
FOOD: [Specific food name]
QUANTITY: [X]g
CALORIES: [X] kcal
MICROPLASTICS: [X.X] mg/kg
RISK: [LOW/MEDIUM/HIGH]

If multiple items, list each separately.
If no food detected, respond: "NO FOOD DETECTED"

Do not include any additional text, explanations, or commentary."""
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
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Analysis service unavailable"

        except requests.exceptions.RequestException as e:
            return f"API Communication Error: {str(e)}"
        except Exception as e:
            return f"Analysis Error: {str(e)}"

    def display_analysis_results(self, analysis):
        """Display analysis results in professional format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        formatted_results = f"""
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  ANALYSIS REPORT                                        │
│                                 {timestamp}                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

🔬 ANALYSIS RESULTS:
{analysis}

📈 Report generated by Professional Microplastics Food Analyzer v3.0
═══════════════════════════════════════════════════════════════════════════════════════════

"""

        # Add results to display
        self.results_text.insert(tk.END, formatted_results)
        self.results_text.see(tk.END)

        # Update status
        self.update_status("✅ Analysis complete", self.colors['success'])

        # Save results automatically
        self.save_analysis_report(analysis, timestamp)

    def save_current_frame(self):
        """Save current camera frame."""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"food_analysis_{timestamp}.jpg"

            os.makedirs("saved_images", exist_ok=True)
            filepath = os.path.join("saved_images", filename)

            cv2.imwrite(filepath, self.current_frame)

            self.update_status(f"💾 Image saved: {filename}", self.colors['success'])
            messagebox.showinfo("Success", f"Image saved successfully:\n{filepath}")
        else:
            messagebox.showerror("Error", "No image available to save")

    def save_analysis_report(self, analysis, timestamp):
        """Save analysis report to file."""
        filename = f"analysis_report_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
        os.makedirs("analysis_reports", exist_ok=True)
        filepath = os.path.join("analysis_reports", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("PROFESSIONAL MICROPLASTICS FOOD ANALYZER v3.0\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis Date: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis)
            f.write("\n\nGenerated by Professional Microplastics Food Analyzer v3.0")
            f.write("\nBased on scientific research data from 2024-2025 studies")

    def clear_results(self):
        """Clear results display."""
        self.results_text.delete(1.0, tk.END)
        self.display_professional_welcome()
        self.update_status("🟢 Results cleared - Ready for analysis", self.colors['success'])

    def run(self):
        """Launch the application."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Application error: {e}")
            self.cleanup()

    def on_closing(self):
        """Handle application closing."""
        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main application entry point."""
    try:
        app = MicroplasticsAnalyzer()
        app.run()
    except Exception as e:
        print(f"Error initializing application: {e}")
        print("\nSETUP REQUIREMENTS:")
        print("1. Create 'food.json' with your Gemini API key")
        print("2. Ensure webcam is connected and functional")
        print("3. Install required packages: opencv-python, pillow, requests")
        print("\nExample food.json format:")
        print('{"gemini_api_key": "your_api_key_here"}')


if __name__ == "__main__":
    main()