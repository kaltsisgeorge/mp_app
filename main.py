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
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Updated CORS for production (we'll update this after frontend deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://*.netlify.app",   # For Netlify deployments
        "https://*.vercel.app",    # For Vercel deployments
        "https://68a33f06b9b6a4425498e9b9--iridescent-mooncake-783006.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodAnalyzer:
    def __init__(self):
        """Initialize the Food Analyzer (without GUI components)."""
        # Try to load from environment variable first, then from file
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            # Fallback to config file for local development
            try:
                self.config = self.load_config('food.json')
                self.gemini_api_key = self.config.get('gemini_api_key')
            except:
                pass
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or provide food.json")
        
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
            
            # Save report (in production, this would be in a temporary directory)
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
        
        # Use /tmp directory in production or create reports directory locally
        report_dir = "/tmp/analysis_reports" if os.path.exists("/tmp") else "analysis_reports"
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
        # Check both possible report directories
        possible_dirs = ["/tmp/analysis_reports", "analysis_reports"]
        report_dir = None
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                report_dir = dir_path
                break
        
        if not report_dir:
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

    def get_analysis_stats(self):
        """Calculate analysis statistics from reports."""
        try:
            reports = self.get_all_reports()
            
            if not reports:
                return {
                    "total_analyses": 0,
                    "total_calories": 0,
                    "total_microplastics": 0,
                    "average_microplastics": 0,
                    "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
                    "recent_analyses": 0,
                    "top_foods": []
                }
            
            total_analyses = len(reports)
            total_calories = 0
            total_microplastics = 0
            risk_counts = {"low": 0, "medium": 0, "high": 0}
            food_counts = {}
            recent_analyses = 0
            
            # Get analyses from last 7 days
            week_ago = datetime.now() - timedelta(days=7)
            
            for report in reports:
                # Parse report content for stats
                content = report["content"]
                
                # Check if analysis is recent
                report_date = datetime.fromisoformat(report["timestamp"])
                if report_date > week_ago:
                    recent_analyses += 1
                
                # Extract data from parsed results section
                lines = content.split('\n')
                current_food = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("FOOD:"):
                        current_food = line.split(":", 1)[1].strip()
                        food_counts[current_food] = food_counts.get(current_food, 0) + 1
                    elif line.startswith("CALORIES:"):
                        try:
                            calories = int(re.findall(r'\d+', line)[0])
                            total_calories += calories
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("MICROPLASTICS:"):
                        try:
                            microplastics = float(re.findall(r'[\d.]+', line)[0])
                            total_microplastics += microplastics
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("RISK:"):
                        risk_level = line.split(":", 1)[1].strip().lower()
                        if risk_level in risk_counts:
                            risk_counts[risk_level] += 1
            
            # Calculate averages
            avg_microplastics = total_microplastics / total_analyses if total_analyses > 0 else 0
            
            # Get top 5 foods
            top_foods = sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_foods = [{"name": food, "count": count} for food, count in top_foods]
            
            return {
                "total_analyses": total_analyses,
                "total_calories": total_calories,
                "total_microplastics": round(total_microplastics, 2),
                "average_microplastics": round(avg_microplastics, 2),
                "risk_distribution": risk_counts,
                "recent_analyses": recent_analyses,
                "top_foods": top_foods
            }
        
        except Exception as e:
            logger.error(f"Error calculating analysis stats: {e}")
            return {
                "total_analyses": 0,
                "total_calories": 0,
                "total_microplastics": 0,
                "average_microplastics": 0,
                "risk_distribution": {"low": 0, "medium": 0, "high": 0},
                "recent_analyses": 0,
                "top_foods": []
            }

    def get_user_stats(self):
        """Get user-related statistics."""
        try:
            stats = self.get_analysis_stats()
            
            # Calculate some user-focused metrics
            total_analyses = stats["total_analyses"]
            avg_calories_per_analysis = stats["total_calories"] / total_analyses if total_analyses > 0 else 0
            
            # Determine user activity level
            if stats["recent_analyses"] >= 10:
                activity_level = "Very Active"
            elif stats["recent_analyses"] >= 5:
                activity_level = "Active"
            elif stats["recent_analyses"] >= 1:
                activity_level = "Moderate"
            else:
                activity_level = "New User"
            
            # Calculate health score based on risk distribution
            risk_dist = stats["risk_distribution"]
            total_risk_items = sum(risk_dist.values())
            if total_risk_items > 0:
                health_score = round(
                    (risk_dist["low"] * 100 + risk_dist["medium"] * 60 + risk_dist["high"] * 20) / total_risk_items
                )
            else:
                health_score = 100
            
            return {
                "total_scans": total_analyses,
                "activity_level": activity_level,
                "average_calories_per_scan": round(avg_calories_per_analysis, 1),
                "health_score": health_score,
                "recent_activity": stats["recent_analyses"],
                "microplastics_exposure": stats["average_microplastics"],
                "favorite_foods": stats["top_foods"][:3]  # Top 3 for user stats
            }
        
        except Exception as e:
            logger.error(f"Error calculating user stats: {e}")
            return {
                "total_scans": 0,
                "activity_level": "New User",
                "average_calories_per_scan": 0,
                "health_score": 100,
                "recent_activity": 0,
                "microplastics_exposure": 0,
                "favorite_foods": []
            }

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

@app.get("/analysis/stats")
async def get_analysis_stats():
    """Get analysis statistics."""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    
    try:
        stats = analyzer.get_analysis_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error fetching analysis stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/stats")
async def get_user_stats():
    """Get user statistics."""
    if not analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    
    try:
        stats = analyzer.get_user_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error fetching user stats: {str(e)}")
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
    print("1. GEMINI_API_KEY environment variable set OR food.json with your Gemini API key")
    print("2. Required packages installed")
    print("3. Frontend running on appropriate port")
    # Use PORT environment variable if available (required for deployment)
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)