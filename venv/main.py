import os
import httpx # A modern, async-compatible HTTP client, like 'fetch' for Python
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# === 1. Create FastAPI App and Configure CORS ===
app = FastAPI()

allowed_origins = [
    "https://andrewilkinson.com",
    "https://www.andrewilkinson.com",
    # Add your local dev URL if you want to test locally
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# === 2. Define the Weather API Endpoint ===
@app.get("/api/weather")
async def get_weather(
    lat: float = Query(..., description="Latitude"), 
    lon: float = Query(..., description="Longitude")
):
    """
    Fetches the current weather for a given latitude and longitude.
    """
    # FastAPI's type hints (lat: float, lon: float) handle the 
    # check for missing parameters automatically.

    weather_api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=fahrenheit"
    
    print(f"Fetching weather for: {lat}, {lon}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(weather_api_url)
            
            # Raise an HTTP exception if the request was not successful
            response.raise_for_status() 
            
            weather_data = response.json()
            
            # --- Send Weather Data to Client ---
            if "current_weather" in weather_data:
                return weather_data["current_weather"]
            else:
                raise HTTPException(status_code=500, detail="Invalid data format from weather API.")

        except httpx.HTTPStatusError as e:
            # Handle errors from the weather API
            print(f"Weather API Error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch weather data: {e.response.text}")
        except Exception as e:
            # Handle other server errors
            print(f"Server Error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error while fetching weather.")

# === 3. Define the Chat API Endpoint ===

# Define a Pydantic model for the request body.
# This ensures the 'prompt' field is present and is a string.
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def post_chat(request: ChatRequest):
    """
    Receives a prompt and sends it to the Google Gemini API.
    """
    api_key = os.environ.get("GEMINI_API_KEY") # Get key from environment variables

    # --- Input Validation ---
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="Server is missing API configuration.")

    # --- Call Google Gemini API ---
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": request.prompt}]}],
        # "systemInstruction": {
        #   "parts": [{"text": "You are a helpful assistant for andrewilkinson.com"}]
        # },
    }
    
    print(f"Sending prompt to LLM: \"{request.prompt[:30]}...\"")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(api_url, json=payload, headers={"Content-Type": "application/json"})
            
            # Raise an exception for bad status codes
            response.raise_for_status()

            result = response.json()
            
            # --- Safely Extract Text and Send to Client ---
            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return {"response": text}
            except (KeyError, IndexError, TypeError):
                print(f"Unexpected API response structure: {result}")
                raise HTTPException(status_code=500, detail="Received an invalid response from the LLM.")

        except httpx.HTTPStatusError as e:
            # Handle errors from the Gemini API
            print(f"Gemini API Error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to get response from LLM: {e.response.text}")
        except Exception as e:
            # Handle other server errors
            print(f"Server Error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error while contacting LLM.")

# Note: The server is started using a command, not from within the file.
