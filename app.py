from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2
from deepface import DeepFace
import tempfile
import os
import shutil
import uvicorn
from pathlib import Path
from typing import Dict, Optional
import uuid
import sys

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    yield
    # Shutdown code

app = FastAPI(lifespan=lifespan)  # Replace your existing app = FastAPI()

# Create required directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/models", exist_ok=True)

app = FastAPI(title="Emotion Detection App")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# User authentication storage (simple dictionary for now)
users = {"admin": "password123"}

# Emoji mapping for emotions
EMOJI_MAP = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😃",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Create templates directory and files
os.makedirs("templates", exist_ok=True)

# Main index.html template - using utf-8 encoding to handle emoji characters
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .tab-content { padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
        .nav-tabs { margin-bottom: 0; }
        .result-section { margin-top: 20px; }
        .emoji { font-size: 2em; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Emotion Detection App</h1>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="register-tab" data-bs-toggle="tab" data-bs-target="#register" type="button" role="tab">📝 Register</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="login-tab" data-bs-toggle="tab" data-bs-target="#login" type="button" role="tab">🔑 Login</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="train-tab" data-bs-toggle="tab" data-bs-target="#train" type="button" role="tab">🛠 Train Model</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="predict-tab" data-bs-toggle="tab" data-bs-target="#predict" type="button" role="tab">🎭 Predict Emotion</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Register Tab -->
            <div class="tab-pane fade show active" id="register" role="tabpanel">
                <h2>🔑 User Registration</h2>
                <form id="registerForm">
                    <div class="mb-3">
                        <label for="registerUsername" class="form-label">Username</label>
                        <input type="text" class="form-control" id="registerUsername" required>
                    </div>
                    <div class="mb-3">
                        <label for="registerPassword" class="form-label">Password</label>
                        <input type="password" class="form-control" id="registerPassword" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Register</button>
                </form>
                <div id="registerStatus" class="alert mt-3" style="display: none;"></div>
            </div>
            
            <!-- Login Tab -->
            <div class="tab-pane fade" id="login" role="tabpanel">
                <h2>🔐 User Login</h2>
                <form id="loginForm">
                    <div class="mb-3">
                        <label for="loginUsername" class="form-label">Username</label>
                        <input type="text" class="form-control" id="loginUsername" required>
                    </div>
                    <div class="mb-3">
                        <label for="loginPassword" class="form-label">Password</label>
                        <input type="password" class="form-control" id="loginPassword" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Login</button>
                </form>
                <div id="loginStatus" class="alert mt-3" style="display: none;"></div>
            </div>
            
            <!-- Train Model Tab -->
            <div class="tab-pane fade" id="train" role="tabpanel">
                <h2>🎓 Train Emotion Model</h2>
                <form id="trainForm">
                    <div class="mb-3">
                        <label for="datasetFile" class="form-label">📂 Upload Dataset (CSV)</label>
                        <input type="file" class="form-control" id="datasetFile" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Train Model</button>
                </form>
                
                <div id="trainingResults" class="result-section" style="display: none;">
                    <div id="trainingStatus"></div>
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <h4>📈 Dataset Overview</h4>
                            <img id="datasetPlot" class="img-fluid" alt="Dataset Plot">
                        </div>
                        <div class="col-md-6">
                            <h4>📊 Top 4 Frames</h4>
                            <pre id="topFrames" class="border p-2"></pre>
                        </div>
                    </div>
                    <div class="mt-3">
                        <h4>📑 Classification Report</h4>
                        <pre id="classReport" class="border p-2"></pre>
                    </div>
                </div>
            </div>
            
            <!-- Predict Emotion Tab -->
            <div class="tab-pane fade" id="predict" role="tabpanel">
                <h2>🎭 Emotion Detection from Video</h2>
                <form id="predictForm">
                    <div class="mb-3">
                        <label for="videoFile" class="form-label">🎥 Upload Video for Emotion Detection</label>
                        <input type="file" class="form-control" id="videoFile" accept="video/*" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Detect Emotion</button>
                </form>
                
                <div id="predictResults" class="result-section" style="display: none;">
                    <div class="row">
                        <div class="col-md-4">
                            <h4>🎭 Predicted Emotion</h4>
                            <div id="predictedEmotion" class="border p-3 text-center"></div>
                        </div>
                        <div class="col-md-8">
                            <h4>📈 Emotion Score Graph</h4>
                            <img id="emotionPlot" class="img-fluid" alt="Emotion Plot">
                        </div>
                    </div>
                    <div class="row mt-3">
                        <div class="col-12">
                            <h4>📈 Video Emotion Trend</h4>
                            <img id="emotionTrend" class="img-fluid" alt="Emotion Trend">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Register form submission
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('registerUsername').value;
            const password = document.getElementById('registerPassword').value;
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        username: username,
                        password: password
                    })
                });
                
                const result = await response.text();
                const statusDiv = document.getElementById('registerStatus');
                statusDiv.textContent = result;
                statusDiv.style.display = 'block';
                statusDiv.className = result.includes('✅') ? 'alert alert-success' : 'alert alert-danger';
            } catch (error) {
                console.error('Error:', error);
            }
        });
        
        // Login form submission
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        username: username,
                        password: password
                    })
                });
                
                const result = await response.text();
                const statusDiv = document.getElementById('loginStatus');
                statusDiv.textContent = result;
                statusDiv.style.display = 'block';
                statusDiv.className = result.includes('✅') ? 'alert alert-success' : 'alert alert-danger';
            } catch (error) {
                console.error('Error:', error);
            }
        });
        
        // Train model form submission
        document.getElementById('trainForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('datasetFile');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/train_model', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Display results
                document.getElementById('trainingStatus').innerHTML = result.status;
                document.getElementById('datasetPlot').src = result.dataset_plot;
                document.getElementById('topFrames').textContent = result.top_frames;
                document.getElementById('classReport').textContent = result.class_report;
                document.getElementById('trainingResults').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
            }
        });
        
        // Predict emotion form submission
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const response = await fetch('/predict_emotion', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }
                
                const result = await response.json();
                
                // Display results
                document.getElementById('predictedEmotion').innerHTML = result.emotion;
                document.getElementById('emotionPlot').src = result.emotion_plot;
                document.getElementById('emotionTrend').src = result.trend_plot;
                document.getElementById('predictResults').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
    """)

# Function to register a new user
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in users:
        return "❌ Username already exists. Please try another."
    users[username] = password
    return "✅ Registration successful! You can now log in."

# Function to login
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        return "✅ Login successful! You can now proceed."
    return "❌ Invalid credentials. Please try again."

# Function to train model and display dataset insights
@app.post("/train_model")
async def train_model(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"static/uploads/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file
    df = pd.read_csv(temp_file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    model_path = "static/models/emotion_model.pkl"
    joblib.dump(model, model_path)
    
    # Display only top 4 rows of dataset
    top_4_data = df.head(4).to_string()
    
    # Plot dataset overview
    plt.figure(figsize=(8, 5))
    df[y.name].value_counts().plot(kind='line', marker='o', color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📊 Dataset Emotion Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = f"static/images/dataset_plot_{uuid.uuid4()}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return {
        "status": f"<h2 style='font-size:24px;'>✅ Model trained successfully with accuracy: {accuracy:.2f} 🎯</h2>",
        "dataset_plot": "/" + plot_path,
        "top_frames": f"📊 Top 4 Rows:\n{top_4_data}",
        "class_report": f"📑 Classification Report:\n{class_report}"
    }

# Function to predict emotion from video and generate line graph
@app.post("/predict_emotion")
async def predict_emotion(video: UploadFile = File(...)):
    # Check if model exists
    model_path = "static/models/emotion_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="⚠ Error: Model not trained yet. Please train the model first.")
    
    # Save the uploaded video
    temp_video_path = f"static/uploads/{video.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Load model
    model = joblib.load(model_path)
    
    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    if duration < 2 or duration > 20:
        cap.release()
        raise HTTPException(status_code=400, detail="⚠ Error: Video duration must be between 2 and 20 seconds. ⏳")
    
    frame_results = {}
    
    frame_interval = int(fps)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                analysis = DeepFace.analyze(img_rgb, actions=["emotion"], enforce_detection=False)
                if analysis:
                    for key, value in analysis[0]['emotion'].items():
                        frame_results[key] = frame_results.get(key, []) + [value]
            except Exception as e:
                cap.release()
                raise HTTPException(status_code=400, detail=f"⚠ Error analyzing video: {str(e)}")
        
        frame_id += 1
    
    cap.release()
    
    if not frame_results:
        raise HTTPException(status_code=400, detail="⚠ Error: Could not detect faces in the video.")
    
    avg_scores = {key: np.mean(values) for key, values in frame_results.items()}
    
    predicted_emotion = max(avg_scores, key=avg_scores.get)
    predicted_emoji = EMOJI_MAP.get(predicted_emotion, "❓")
    
    # Plot emotions as bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(list(avg_scores.keys()), list(avg_scores.values()), color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📊 Emotion Analysis Over Video")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    emotion_plot_path = f"static/images/emotion_plot_{uuid.uuid4()}.png"
    plt.savefig(emotion_plot_path)
    plt.close()
    
    # Generate line graph for emotions
    plt.figure(figsize=(10, 5))
    plt.plot(list(avg_scores.keys()), list(avg_scores.values()), marker='o', linestyle='-')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("📈 Emotion Trends Over Video")
    plt.grid()
    
    trend_plot_path = f"static/images/video_trend_plot_{uuid.uuid4()}.png"
    plt.savefig(trend_plot_path)
    plt.close()
    
    return {
        "emotion": f"{predicted_emotion} {predicted_emoji}",
        "emotion_plot": "/" + emotion_plot_path,
        "trend_plot": "/" + trend_plot_path
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Create a requirements.txt file
with open("requirements.txt", "w") as f:
    f.write("""fastapi==0.104.1
uvicorn==0.23.2
python-multipart==0.0.6
jinja2==3.1.2
pandas==2.1.1
numpy==1.26.0
matplotlib==3.8.0
scikit-learn==1.3.1
joblib==1.3.2
opencv-python==4.8.1.78
deepface==0.0.79
""")

# Create a Procfile for Render - also using utf-8 encoding
with open("your_file.txt", "w", encoding="utf-8") as f:
    f.write("web: uvicorn main:app --host=127.0.0.1 --port=${PORT:-8080}")

if __name__ == "main":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)

