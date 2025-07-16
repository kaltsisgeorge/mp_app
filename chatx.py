import tkinter as tk
from tkinter import ttk

class MicroplasticsFoodAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Microplastics Food Analyzer")
        self.root.geometry("600x400")

        self.colors = {
            'purple_medium': '#6A5ACD',  # Medium Slate Blue
            'text_light': '#FFFFFF',     # White
            'background': '#2E2E2E'      # Dark background
        }

        # Set root background color
        self.root.configure(bg=self.colors['background'])

        self.style = ttk.Style()
        self.style.theme_use('default')

        # Copy layout from default so the custom style exists without error
        self.style.layout('Purple.TLabelFrame', self.style.layout('TLabelFrame'))
        self.style.layout('Purple.TLabelFrame.Label', self.style.layout('TLabelFrame.Label'))

        # Configure the purple LabelFrame style
        self.style.configure('Purple.TLabelFrame',
                             background=self.colors['purple_medium'],
                             foreground=self.colors['text_light'],
                             borderwidth=2,
                             relief='solid')

        self.style.configure('Purple.TLabelFrame.Label',
                             background=self.colors['purple_medium'],
                             foreground=self.colors['text_light'],
                             font=('Segoe UI', 14, 'bold'))

        # Create a LabelFrame with the purple style
        self.frame = ttk.LabelFrame(self.root, text="Food Microplastics Data", style='Purple.TLabelFrame')
        self.frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Example content inside the frame
        label = ttk.Label(self.frame, text="Analyze your food samples here.", background=self.colors['purple_medium'], foreground=self.colors['text_light'])
        label.pack(pady=10)

        self.root.mainloop()

if __name__ == "__main__":
    app = MicroplasticsFoodAnalyzer()
