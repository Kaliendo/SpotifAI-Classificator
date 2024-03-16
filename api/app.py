import os
import joblib
import uvicorn
import requests
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sklearn.tree import DecisionTreeClassifier

load_dotenv()

SPOTIFY_TOKEN = None
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

class SongFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

class SpotifyURL(BaseModel):
    url: str

app = FastAPI()

model = joblib.load("model.pkl")

def get_token():
    global SPOTIFY_TOKEN
    data = {
        'grant_type': 'client_credentials',
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET,
    }

    url = "https://accounts.spotify.com/api/token"
    response = requests.post(url, data=data)

    SPOTIFY_TOKEN = response.json()["access_token"]

def get_spotify_track_id(spotify_url: str) -> str:
    # parts = spotify_url.split('/track/')[0]
    # track_id = parts[-1].split('?')[0]
    return spotify_url.split('/track/')[1]

def fetch_track_features(track_id: str):
    global SPOTIFY_TOKEN
    endpoint = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = {
        "Authorization": f"Bearer {SPOTIFY_TOKEN}"
    }
    response = requests.get(endpoint, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch track features from Spotify")
    return response.json()

@app.post("/analyze-manual")
async def make_prediction(song: SongFeatures):
    try:
        input_data = np.array([[song.danceability, song.energy, song.key, song.loudness, song.speechiness,
                                song.acousticness, song.instrumentalness, song.liveness, song.valence, song.tempo]])
        
        prediction = model.predict(input_data)
        
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/analyze-url")
async def predict_from_spotify(spotify_url: SpotifyURL):
    try:
        track_id = get_spotify_track_id(spotify_url.url)
        features = fetch_track_features(track_id)
        input_data = np.array([[features['danceability'], features['energy'], features['key'], features['loudness'],
                                features['speechiness'], features['acousticness'], features['instrumentalness'],
                                features['liveness'], features['valence'], features['tempo']]])
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.on_event("startup")
async def startup_event():
    get_token()

if __name__ == "__main__":
    print(get_token())
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)