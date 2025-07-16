from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from PIL import Image
from typing import List, Dict, Optional
import json
import io
import base64
import numpy as np
import requests
import os
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodAnalyzer:
    def __init__(self):
        """Initialize the Food Analyzer (without GUI components)."""
        self.config = self.load_config('food.json')
        self.gemini_api_key = self.config.get('gemini_api_key')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found in config")
        
        # API Configuration
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        
        logger.info("Food Analyzer initialized successfully (API mode)")

    def load_config(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file {config_file}")

    def encode_image(self, image_array):
        """Encode image array to base64 for API submission."""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array)
            else:
                image = image_array
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

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
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                raise ValueError("No analysis results received from Gemini")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def analyze_food_image(self, image_array):
        """Main function to analyze food image."""
        try:
            # Encode image
            image_base64 = self.encode_image(image_array)
            
            # Get analysis from Gemini
            analysis_text = self.analyze_with_gemini(image_base64)
            
            # Parse results
            parsed_results = self.parse_analysis_results(analysis_text)
            
            # Save report
            report_path = self.save_analysis_report(parsed_results, analysis_text, image_array)
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "raw_analysis": analysis_text,
                "parsed_results": parsed_results,
                "report_path": report_path
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def parse_analysis_results(self, analysis_text):
        """Parse the analysis text into structured data."""
        if analysis_text == "NO FOOD DETECTED":
            return []
            
        results = []
        current_item = {}
        
        for line in analysis_text.split('\n'):
            if not line.strip():
                if current_item:
                    results.append(current_item)
                    current_item = {}
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Remove brackets if present
                value = value.strip('[]')
                
                if key in ['food', 'quantity', 'calories', 'microplastics', 'risk']:
                    current_item[key] = value
        
        if current_item:
            results.append(current_item)
            
        return results

    def save_analysis_report(self, analysis_data, raw_analysis, image_array=None):
        """Save analysis results to a report file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = "analysis_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"analysis_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"Food Analysis Report - {timestamp}\n")
            f.write("-" * 40 + "\n\n")
            f.write(f"RAW ANALYSIS:\n{raw_analysis}\n\n")
            f.write("PARSED RESULTS:\n")
            
            for item in analysis_data:
                for key, value in item.items():
                    f.write(f"{key.upper()}: {value}\n")
                f.write("\n")
        
        return report_path

    def get_all_reports(self, limit: Optional[int] = None):
        """Get all saved analysis reports."""
        report_dir = "analysis_reports"
        if not os.path.exists(report_dir):
            return []
        
        # Get all report files
        report_files = glob.glob(os.path.join(report_dir, "analysis_report_*.txt"))
        
        # Sort by creation time (newest first)
        report_files.sort(key=os.path.getctime, reverse=True)
        
        # Apply limit if specified
        if limit:
            report_files = report_files[:limit]
        
        reports = []
        for file_path in report_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract timestamp from filename
                filename = os.path.basename(file_path)
                timestamp_str = filename.replace("analysis_report_", "").replace(".txt", "")
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                except ValueError:
                    timestamp = datetime.fromtimestamp(os.path.getctime(file_path))
                
                reports.append({
                    "id": timestamp_str,
                    "timestamp": timestamp.isoformat(),
                    "created": timestamp.isoformat(),
                    "content": content,
                    "file_path": file_path
                })
            except Exception as e:
                logger.error(f"Error reading report {file_path}: {e}")
                continue
        
        return reports

# Initialize the analyzer
try:
    analyzer = FoodAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize analyzer: {e}")
    analyzer = None

@app.get("/")
async def root():
    return {"message": "Food Analysis API is running"}

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
    try:
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Analyze the image
        result = analyzer.analyze_food_image(image)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports")
async def get_reports(limit: Optional[int] = 50):
    """Get all saved analysis reports."""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    
    try:
        reports = analyzer.get_all_reports(limit=limit)
        return {
            "success": True,
            "count": len(reports),
            "reports": reports
        }
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if analyzer else "analyzer_not_initialized",
        "analyzer_initialized": analyzer is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Food Analysis API...")
    print("Make sure you have:")
    print("1. food.json with your Gemini API key")
    print("2. Required packages installed")
    print("3. Frontend running on http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=8002)