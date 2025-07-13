# 🧠 NLP Sentiment Analysis App — React + FastAPI + Docker

Welcome to **NLP Sentiment Analysis**, a full-stack AI-powered web application that detects the sentiment of user-submitted text using a fine-tuned Hugging Face model. It's fast, fun, Dockerized, and made for showcasing AI + web dev chops!

![Sentiment AI](https://img.shields.io/badge/NLP-Sentiment-blueviolet) ![React](https://img.shields.io/badge/Frontend-React-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![Docker](https://img.shields.io/badge/Deployment-Docker-orange) ![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)

---

## 🚀 Tech Stack

| Layer         | Tech Used                     |
|---------------|-------------------------------|
| Frontend      | React (with create-react-app) |
| Backend       | FastAPI + Uvicorn             |
| Model         | Hugging Face Transformers     |
| Fine-tuning   | PyTorch + Datasets            |
| Deployment    | Docker + Docker Compose       |

---

## 📁 Project Structure

```bash
nlp-sentiment-react-fastapi/
├── backend/
│   ├── app.py                 # FastAPI backend logic
│   ├── finetune.py            # CLI script for model fine-tuning
│   ├── model/                 # Folder where trained model is saved
│   ├── data/                  # Sample training data (JSONL)
│   ├── Dockerfile             # Backend Dockerfile
│   └── requirements.txt       # Python dependencies
│
├── frontend/
│   ├── src/                   # React source files
│   ├── public/                # Static assets
│   ├── Dockerfile             # Frontend Dockerfile
│   └── package.json           # Frontend config
│
├── docker-compose.yml         # Orchestrates frontend + backend
└── README.md                  # You're reading it 😎
```

---

## 🛠️ How to Run (Dockerized)

> Make sure Docker & Docker Compose are installed

```bash
git clone https://github.com/vishnudaspk/nlp-sentiment-react-fastapi.git
cd nlp-sentiment-react-fastapi

docker-compose up --build
```

- Frontend 👉 http://localhost:3000
- Backend API 👉 http://localhost:8000
- Swagger docs 👉 http://localhost:8000/docs

---
## 🐳 Run the Backend via Docker (Prebuilt Image)

Want to test the backend without cloning or building anything?

Just pull the image from Docker Hub and run it:

```bash
docker pull vishnu71y13/sentiment-backend:latest
docker run -p 8000:8000 vishnu71y13/sentiment-backend


## 📡 API Endpoint

**POST `/predict`**
```json
Request:
{
  "text": "I love this app!"
}

Response:
{
  "label": "positive",
  "score": 0.9843
}
```
Test it via Swagger at `/docs` or from the frontend UI.

---

## 🧪 Model Fine-Tuning

To fine-tune your own sentiment model:
```bash
python backend/finetune.py \
  --data backend/data/data.jsonl \
  --epochs 3 \
  --lr 3e-5
```

Model will be saved
