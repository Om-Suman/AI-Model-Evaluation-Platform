# 🤖 AI Model Evaluation Platform

A production-ready platform for evaluating Large Language Models (LLMs) on custom datasets.
Upload a question-answer CSV, select a model, run an automated evaluation pipeline, and compare
performance across models using BLEU, ROUGE, Accuracy, and Latency metrics.

---

## 📁 Project Structure

```
ai_model_eval_platform/
├── backend/
│   ├── main.py               # FastAPI app factory + CORS + lifespan startup
│   ├── database.py           # MySQL connection helper + schema DDL + seeding
│   ├── models.py             # Pydantic DTOs for request/response validation
│   ├── evaluation_engine.py  # End-to-end LLM evaluation orchestrator
│   ├── metrics.py            # BLEU, ROUGE, accuracy, latency calculations
│   └── routes.py             # All FastAPI route handlers
├── frontend/
│   └── app.py                # Streamlit multi-page dashboard
├── datasets/
│   └── sample_qa.csv         # Sample 15-question dataset to get started
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Tech Stack

| Layer      | Technology                          |
|------------|-------------------------------------|
| Backend    | FastAPI + Uvicorn                   |
| Frontend   | Streamlit + Plotly                  |
| Database   | MySQL 8+                            |
| LLM APIs   | Together.ai, HuggingFace Inference  |
| ML / NLP   | scikit-learn, evaluate, NLTK        |
| Data       | pandas, numpy                       |
| ORM / DB   | mysql-connector-python              |

---

## 🚀 Setup Instructions

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd ai_model_eval_platform
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some NLTK data is required for BLEU scoring:

```python
python -c "import nltk; nltk.download('punkt')"
```

### 4. Set up MySQL

Ensure MySQL 8+ is running, then create the application user:

```sql
CREATE USER 'eval_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ai_eval_platform.* TO 'eval_user'@'localhost';
FLUSH PRIVILEGES;
```

> The application will auto-create the `ai_eval_platform` database and all
> tables on first startup — you do **not** need to run any migration scripts.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=eval_user
DB_PASSWORD=your_password
DB_NAME=ai_eval_platform

TOGETHER_API_KEY=<your Together.ai key>   # https://api.together.xyz
HUGGINGFACE_API_KEY=<your HF token>       # https://huggingface.co/settings/tokens

BACKEND_URL=http://localhost:8000
```

### 6. Start the FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 7. Start the Streamlit dashboard

Open a **second terminal** (keep the backend running):

```bash
cd frontend
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

---

## 🗄️ Database Schema

```
models          – LLM catalogue (pre-seeded with Llama, Mistral, Gemma)
datasets        – Uploaded CSV metadata (name, filename, row_count)
evaluations     – One row per evaluation run (status, timestamps)
metrics         – Aggregate scores per evaluation (accuracy, BLEU, ROUGE, latency)
```

---

## 🔌 API Endpoints

| Method | Path                              | Description                        |
|--------|-----------------------------------|------------------------------------|
| GET    | `/health`                         | Liveness check                     |
| GET    | `/models`                         | List all available LLMs            |
| GET    | `/datasets`                       | List all uploaded datasets         |
| POST   | `/upload-dataset`                 | Upload a CSV dataset               |
| POST   | `/evaluate-model`                 | Run an evaluation (sync)           |
| GET    | `/evaluation-results`             | List all past evaluation runs      |
| GET    | `/evaluation-results/{id}`        | Detail for a single run            |

---

## 📊 Metrics Explained

| Metric        | Description                                                   |
|---------------|---------------------------------------------------------------|
| **Accuracy**  | Case-insensitive exact match between model answer and ground truth |
| **BLEU**      | N-gram overlap score (sacrebleu corpus-level, normalised 0–1) |
| **ROUGE-1**   | Unigram overlap F1 between prediction and reference           |
| **ROUGE-2**   | Bigram overlap F1                                             |
| **ROUGE-L**   | Longest common subsequence F1                                 |
| **Avg Latency** | Mean API response time in milliseconds                      |

---

## 🤖 Supported Models

| Display Name   | Provider      | Model ID                                    |
|----------------|---------------|---------------------------------------------|
| Llama-3 8B     | Together.ai   | `meta-llama/Llama-3-8b-chat-hf`             |
| Llama-3 70B    | Together.ai   | `meta-llama/Llama-3-70b-chat-hf`            |
| Mistral 7B     | Together.ai   | `mistralai/Mistral-7B-Instruct-v0.2`        |
| Mixtral 8x7B   | Together.ai   | `mistralai/Mixtral-8x7B-Instruct-v0.1`      |
| Gemma 7B       | Together.ai   | `google/gemma-7b-it`                        |
| Gemma 2B       | HuggingFace   | `google/gemma-2b-it`                        |

You can add more models directly to the `models` table in MySQL.

---

## 📄 Dataset Format

CSV files must contain exactly these two columns:

```csv
question,ground_truth_answer
What is the capital of France?,Paris
Who wrote Hamlet?,William Shakespeare
```

A 15-question sample file is included at `datasets/sample_qa.csv`.

---

## 🛠️ Extending the Platform

- **Add a new model**: Insert a row into the `models` table with the correct `provider` and `model_id`.
- **Add a new metric**: Extend `metrics.py` and add the column to `_CREATE_METRICS` DDL in `database.py`.
- **Async evaluations**: Wrap `run_evaluation` with FastAPI `BackgroundTasks` or Celery for non-blocking runs.
- **Auth**: Add an OAuth2 dependency to FastAPI routes for multi-user support.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Streamlit | Start the FastAPI backend first |
| `Access denied` for MySQL | Check `.env` credentials; grant privileges |
| `TOGETHER_API_KEY` errors | Sign up at https://api.together.xyz and add key to `.env` |
| Missing CSV columns | Ensure your CSV has `question` and `ground_truth_answer` columns |
| Evaluation times out | Reduce dataset size or increase the Streamlit request timeout |
