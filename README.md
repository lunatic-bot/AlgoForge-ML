# AlgoForge-ML

AlgoForge is an end-to-end, API-driven machine learning platform that allows users to dynamically build, train, evaluate, and test models in real-time. Designed with a strict microservice architecture, it cleanly separates the core mathematical logic, backend routing, and user interface, mirroring enterprise-grade MLOps systems.

## ✨ Key Features

- **Strict Separation of Concerns:** Core ML logic is decoupled from the web layer.
- **RESTful API Backend:** High-performance, asynchronous endpoints powered by FastAPI.
- **Interactive Dashboard:** Beautiful, reactive frontend built with Streamlit.
- **Custom CSV Processing:** Upload real-world, messy datasets. The backend automatically handles missing value imputation (median) and One-Hot Encoding for text categories.
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) visualizes exactly _why_ the model made a specific prediction on the frontend.
- **Auto-Tuning:** Toggle built-in `GridSearchCV` to automatically find the mathematical optimum hyperparameters for any algorithm.
- **Model Registry & Persistence:** Models are securely saved to disk using `joblib` alongside lightweight JSON metadata, surviving server restarts and allowing users to browse their training history.
- **Smart Task Routing:** Automatically detects and formats UI/API responses based on whether the task is Classification (discrete labels) or Regression (continuous floats).

## 🛠️ Technology Stack

**Backend & API**

- **FastAPI:** High-performance web framework for the API layer.
- **Uvicorn:** Lightning-fast ASGI server.
- **Pydantic:** Strict data validation and schema definition.

**Machine Learning Engine**

- **Scikit-Learn:** Core algorithm implementations and hyperparameter tuning.
- **Pandas & NumPy:** Data ingestion, preprocessing, and tensor manipulation.
- **SHAP:** Explainable AI and feature impact calculations.
- **Joblib:** High-performance model serialization and disk storage.

**Frontend & Testing**

- **Streamlit:** Rapid UI prototyping and interactive data dashboards.
- **Pytest & HTTPX:** Comprehensive integration testing for core ML logic and API contracts.

## 🏗️ Architecture

```text
AlgoForge-ML/
├── core/                   # 🧠 The ML Engine
│   ├── base_model.py       # Abstract Base Class (Auto-handles GridSearchCV)
│   ├── models/             # Concrete model implementations
│   │   ├── tree_models.py  # RandomForestRunner
│   │   ├── linear_models.py# LogisticRegressionRunner, SVRRunner
│   │   └── distance.py     # KNNRunner, SVMRunner
│   └── data_loader.py      # Imputation, Encoding, Scaling, & Splitting
│
├── api/                    # ⚙️ The FastAPI Backend
│   ├── main.py             # FastAPI app with CORS middleware
│   ├── routes.py           # /train, /predict, /upload, /models/history
│   └── schemas.py          # Strict Pydantic contracts
│
├── frontend/               # 🖥️ The Streamlit UI
│   ├── app.py              # Main Streamlit entry point
│   └── components.py       # Reusable UI components
│
├── data/                   # 📊 Local Datasets
│   ├── raw/                # Uploaded custom CSVs
│   └── processed/          # Cleaned data ready for training
│
├── models/                 # 💾 Persistent Storage
│   └── saved/              # .joblib weights and _meta.json history files
│
└── tests/                  # 🧪 Test Suite
    ├── conftest.py         # Pytest anchor
    ├── test_core.py        # Tests data processing logic
    └── test_api.py         # Tests endpoint integration
```

## 📦 Current Features

### Supported Models

| Model                          | Task Type      | Requires Scaling |
| ------------------------------ | -------------- | ---------------- |
| Random Forest                  | Classification | ❌ No            |
| Logistic Regression            | Classification | ✅ Yes           |
| Support Vector Regressor (SVR) | Regression     | ✅ Yes           |
| K-Nearest Neighbors (KNN)      | Classification | ✅ Yes           |
| Support Vector Machine (SVM)   | Classification | ✅ Yes           |

### Built-in Datasets

- **Iris Dataset** - Multiclass classification (3 flower types)
- **Breast Cancer Dataset** - Binary classification (Malignant/Benign)
- **Diabetes Dataset** - Regression (Disease progression)
- **Custom Uploads** - Any valid CSV file

### API Endpoints

| Endpoint          | Method | Description                                   |
| ----------------- | ------ | --------------------------------------------- |
| `/datasets`       | GET    | List built-in datasets                        |
| `/models`         | GET    | List available model algorithms               |
| `/models/history` | GET    | Retrieve metadata for all saved models        |
| `/upload`         | POST   | Upload custom CSV and parse headers           |
| `/train`          | POST   | Train (or auto-tune) and save to disk         |
| `/predict`        | POST   | Load model, predict, and generate SHAP values |

## 🛠️ Installation & Testing

```bash
# Clone the repository
cd AlgoForge-ML

# Install dependencies
pip install -r requirements.txt

# Run the Test Suite to verify logic
python -m pytest -v -s

# Start the API server (Terminal 1)
uvicorn api.main:app --reload

# Start the Streamlit frontend (Terminal 2)
streamlit run frontend/app.py
```

Then open:

- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501

## 🔄 Current Workflow

1. **Upload or Select Data:** Use built-in datasets or upload a custom CSV.
2. **Feature Selection:** Use the dynamic dropdowns to drop noisy columns (e.g., Names, IDs).
3. **Configure & Tune:** Choose an algorithm and toggle `GridSearchCV` for auto-tuning.
4. **Train:** The API handles imputation, encoding, and scaling, then saves the `.joblib` to disk.
5. **View History:** Check the Model Registry to see past model performance metrics.
6. **Predict & Explain:** Pass JSON features to the model to get predictions alongside a SHAP Feature Importance chart explaining the decision.

## 📅 Future Plans

### Phase 5: Cloud Deployment & Containerization

- [ ] Containerize backend and frontend with Docker (`docker-compose.yml`)
- [ ] Deploy API to cloud platforms (Render, Railway, or AWS ECS)
- [ ] Set up CI/CD with GitHub Actions for automated `pytest` runs
- [ ] Add environment variables for production API URL routing

### Phase 6: Advanced ML Upgrades

- [ ] Add XGBoost and LightGBM model runners
- [ ] Export predictions to CSV from the frontend
- [ ] Add learning curves and cross-validation visualization dashboards
- [ ] Add `RandomForestRegressor` and standard `LinearRegression`

## 📄 License

MIT License - Feel free to use this project for learning or building your own ML applications!

---

_Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-Learn._
