import os
import joblib
import uvicorn
import requests
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sklearn.tree import DecisionTreeClassifier
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        return {"genre": prediction[0],
                "features":{
                    "danceability": song.danceability,
                    "energy": song.energy,
                    "key": song.key,
                    "loudness": song.loudness,
                    "speechiness": song.speechiness,
                    "acousticness": song.acousticness,
                    "instrumentalness": song.instrumentalness,
                    "liveness": song.liveness,
                    "valence": song.valence,
                    "tempo": song.tempo,
                    }
                }
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
        return {"genre": prediction[0],
                "features":{
                    "danceability": features['danceability'],
                    "energy": features['energy'],
                    "key": features['key'],
                    "loudness": features['loudness'],
                    "speechiness": features['speechiness'],
                    "acousticness": features['acousticness'],
                    "instrumentalness": features['instrumentalness'],
                    "liveness": features['liveness'],
                    "valence": features['valence'],
                    "tempo": features['tempo'],
                    }
                }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.on_event("startup")
async def startup_event():
    get_token()

if __name__ == "__main__":
    print(get_token())
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)