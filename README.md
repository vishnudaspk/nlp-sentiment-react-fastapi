# 🧠 NLP Sentiment Analysis App — React + FastAPI + Docker

Welcome to **NLP Sentiment Analysis**, a full-stack AI-powered web application that detects the sentiment of user-submitted text using a fine-tuned Hugging Face model. It's fast, fun, Dockerized, and made for showcasing AI + web dev chops!

![Sentiment AI](https://img.shields.io/badge/NLP-Sentiment-blueviolet) ![React](https://img.shields.io/badge/Frontend-React-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![Docker](https://img.shields.io/badge/Deployment-Docker-orange) ![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)

---

## 🚀 Tech Stack

| Layer       | Tech Used                     |
| ----------- | ----------------------------- |
| Frontend    | React (with create-react-app) |
| Backend     | FastAPI + Uvicorn             |
| Model       | Hugging Face Transformers     |
| Fine-tuning | PyTorch + Datasets            |
| Deployment  | Docker + Docker Compose       |

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

* Frontend 👉 [http://localhost:3000](http://localhost:3000)
* Backend API 👉 [http://localhost:8000](http://localhost:8000)
* Swagger docs 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Run the Backend via Docker Hub

Don't want to clone and build locally?

```bash
docker pull vishnu71y13/sentiment-backend:latest
docker run -p 8000:8000 vishnu71y13/sentiment-backend
```

Test it at: `http://localhost:8000/docs`

---

## 📱 API Usage — FastAPI Swagger Docs

Once running, explore the API here: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🧪 Example Request (POST /predict)

```json
{
  "text": "The product exceeded my expectations!"
}
```

### ✅ Example Response

```json
{
  "label": "POSITIVE",
  "score": 0.9864
}
```

---

## 🤪 Model Fine-Tuning

To fine-tune your own sentiment model:

```bash
python backend/finetune.py \
  --data backend/data/data.jsonl \
  --epochs 3 \
  --lr 3e-5
```

Model will be saved to the `backend/model/` directory.

---

## Suggestions to Improve Confidence & Model Performance

If you want to enhance your sentiment analysis model further or ensure it generalizes well, here are some tips and best practices you can consider:

###  1. Add More Diverse Data

* **Why**: The model might be overfitting if trained on a small dataset.
* **What to Do**: Incorporate more real-world text examples across domains (e.g., reviews, tweets, feedback).

###  2. Perform Error Analysis

* **Why**: Helps you understand where the model is failing.
* **What to Do**: Manually inspect misclassified samples and edge cases.

###  3. Use k-Fold Cross Validation

* **Why**: Gives a better generalization estimate compared to a single train-test split.
* **What to Do**: Implement `StratifiedKFold` or `KFold` using scikit-learn during training.

###  4. Try Simpler Models First

* **Why**: Small datasets might work well with Logistic Regression, SVMs, or Naive Bayes.
* **What to Do**: Run a baseline with simpler models before jumping to Transformers.

###  5. Combine Human + Model Evaluation

* **Why**: Human judgment adds valuable context.
* **What to Do**: Validate predictions manually alongside metrics.

###  6. Regularly Validate Model Drift

* **Why**: Sentiment changes over time, especially on dynamic platforms like social media.
* **What to Do**: Periodically fine-tune your model on new, recent data.


> ⚠️ These practices aren't mandatory for this project to work, but if you're planning to take this into production, apply it to a larger corpus, or simply want to experiment more rigorously, these suggestions will help!
---


## 📅 Author & Credits

Built with ❤️ by [Vishnu Das P K](https://github.com/vishnudaspk)
Docker Image: [vishnu71y13/sentiment-backend](https://hub.docker.com/r/vishnu71y13/sentiment-backend)

---

Happy Coding! 🤖
