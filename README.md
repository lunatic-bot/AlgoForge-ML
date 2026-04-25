# AlgoForge-ML

AlgoForge is an end-to-end, API-driven machine learning platform that allows users to dynamically build, train, evaluate, and test models in real-time. Designed with a strict microservice architecture, it cleanly separates the core mathematical logic, backend routing, and user interface, mirroring enterprise-grade MLOps systems.

## ✨ Key Features

- **Strict Separation of Concerns:** Core ML logic is decoupled from the web layer.
- **RESTful API Backend:** High-performance, asynchronous endpoints powered by FastAPI.
- **Interactive Dashboard:** Beautiful, reactive frontend built with Streamlit.
- **Dynamic Form Generation:** Automatically adapts prediction input fields based on the dataset the model was trained on.
- **Human-Readable Outputs:** Built-in translation layers to map mathematical predictions (e.g., `0`, `1`) back to human-readable class labels (e.g., `Setosa`, `Malignant`).
- **State Management:** In-memory registry to handle model persistence, scalers, and label encoders across user sessions.

## 🛠️ Technology Stack

**Backend & API**

- **FastAPI:** High-performance web framework for the API layer.
- **Uvicorn:** Lightning-fast ASGI server.
- **Pydantic:** Strict data validation and schema definition.

**Machine Learning Engine**

- **Scikit-Learn:** Core algorithm implementations (Trees, Linear Models, Distance Models).
- **Pandas & NumPy:** Data ingestion, preprocessing, and tensor manipulation.

**Frontend**

- **Streamlit:** Rapid UI prototyping and interactive data dashboards.
- **Requests:** HTTP library for API communication.

## 🏗️ Architecture

```
AlgoForge-ML/
├── core/                   # 🧠 The ML Engine
│   ├── base_model.py       # Abstract Base Class (BaseMLModel)
│   ├── models/             # Concrete model implementations
│   │   ├── tree_models.py  # RandomForestRunner
│   │   ├── linear_models.py# LogisticRegressionRunner
│   │   └── distance.py     # KNNRunner, SVMRunner
│   └── data_loader.py      # Data preprocessing & splitting
│
├── api/                    # ⚙️ The FastAPI Backend
│   ├── main.py             # FastAPI app with CORS middleware
│   ├── routes.py           # /train, /predict, /datasets endpoints
│   └── schemas.py          # Pydantic models for validation
│
├── frontend/               # 🖥️ The Streamlit UI
│   ├── app.py              # Main Streamlit entry point
│   └── components.py       # Reusable UI components
│
├── data/                   # 📊 Local Datasets
│   ├── raw/                # Unmodified CSVs
│   └── processed/          # Cleaned data ready for training
│
└── tests/                  # 🧪 Unit Tests
    ├── test_models.py      # ML class tests
    └── test_api.py         # API endpoint tests
```

## 📦 Current Features

### Supported Models (Classification Only)

| Model                        | Type           | Requires Scaling |
| ---------------------------- | -------------- | ---------------- |
| Random Forest                | Tree-based     | ❌ No            |
| Logistic Regression          | Linear         | ✅ Yes           |
| K-Nearest Neighbors (KNN)    | Distance-based | ✅ Yes           |
| Support Vector Machine (SVM) | Distance-based | ✅ Yes           |

### Built-in Datasets

- **Iris Dataset** - Multiclass classification (3 flower types)
- **Breast Cancer Dataset** - Binary classification (malignant/benign)

### API Endpoints

| Endpoint    | Method | Description                         |
| ----------- | ------ | ----------------------------------- |
| `/datasets` | GET    | List available datasets             |
| `/models`   | GET    | List available model types          |
| `/train`    | POST   | Train a model on selected dataset   |
| `/predict`  | POST   | Make predictions with trained model |

### Data Processing

- Automatic train/test splitting
- One-hot encoding for categorical features
- Standard scaling for distance/linear models
- Label encoding for target variables

## 🛠️ Installation

```bash
# Clone the repository
cd AlgoForge-ML

# Install dependencies
pip install -r requirements.txt

# Start the API server (Terminal 1)
uvicorn api.main:app --reload

# Start the Streamlit frontend (Terminal 2)
streamlit run frontend/app.py
```

Then open:

- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501

## 🔄 Current Workflow

1. **Select a dataset** (Iris or Breast Cancer)
2. **Choose a model** (Random Forest, Logistic Regression, KNN, or SVM)
3. **Configure hyperparameters** (optional)
4. **Train** - The API handles preprocessing automatically
5. **Evaluate** - Get accuracy scores and detailed reports
6. **Predict** - Use the trained model for new predictions

## 📅 Future Plans

### Phase 1: Regression Support

- [ ] Add `RandomForestRegressor`, `LinearRegressionRunner`, `Ridge`, `Lasso`
- [ ] Implement regression metrics (MSE, RMSE, R², MAE)
- [ ] Add housing price dataset for regression testing
- [ ] Update API to support `task_type` parameter (classification/regression)

### Phase 2: Cloud Deployment

- [ ] Containerize with Docker
- [ ] Deploy to cloud platforms:
  - **Render** (free tier friendly)
  - **Railway**
  - **AWS ECS / Google Cloud Run**
- [ ] Set up CI/CD with GitHub Actions
- [ ] Add environment variables for production settings

### Phase 3: Enhanced Features

- [ ] Add more datasets (Titanic, MNIST, custom CSV upload)
- [ ] Model persistence (save/load trained models)
- [ ] Hyperparameter tuning UI
- [ ] Model comparison dashboard
- [ ] Export predictions to CSV

### Phase 4: Advanced ML

- [ ] Add XGBoost and LightGBM support
- [ ] Feature selection techniques
- [ ] Cross-validation implementation
- [ ] Learning curves visualization

## 📄 License

MIT License - Feel free to use this project for learning or building your own ML applications!

---

Built with ❤️ using Python and scikit-learn
