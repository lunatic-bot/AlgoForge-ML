import pandas as pd
from sklearn.datasets import load_breast_cancer

# Import our custom architecture
from core.data_loader import MLDataLoader
from core.models.tree_models import RandomForestRunner
from core.models.linear_models import LogisticRegressionRunner
from core.models.distance import KNNRunner, SVMRunner

def main():
    print("🚀 Starting Core ML Engine Test...")

    # 1. Load a sample dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target # 0: malignant, 1: benign

    # 2. Initialize the Data Loader
    loader = MLDataLoader(target_column='target', test_size=0.2)

    # 3. Process Data
    # We generate two sets: Scaled (for Distance/Linear) and Raw (for Trees)
    print("📊 Processing Data...")
    X_train_scaled, X_test_scaled, y_train, y_test = loader.process_data(df, requires_scaling=True)
    X_train_raw, X_test_raw, _, _ = loader.process_data(df, requires_scaling=False)

    # 4. Define the models we want to test along with their scaling requirement
    # Format: "Name": (Model Instance, Needs Scaling)
    models = {
        "Random Forest": (RandomForestRunner(n_estimators=50, random_state=42), False),
        "Logistic Regression": (LogisticRegressionRunner(max_iter=1000, random_state=42), True),
        "K-Nearest Neighbors": (KNNRunner(n_neighbors=5), True),
        "Support Vector Machine": (SVMRunner(kernel='linear', C=1.0, random_state=42), True)
    }

    # 5. Train and Evaluate Loop
    print("\n⚙️ Training and Evaluating Models:\n" + "-"*35)
    for name, (model, needs_scaling) in models.items():
        print(f"-> Running {name}...")
        
        # Feed the correct data based on the algorithm's mathematical needs
        X_train = X_train_scaled if needs_scaling else X_train_raw
        X_test = X_test_scaled if needs_scaling else X_test_raw

        # Execute our standardized contract
        model.train(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
        # Extract metrics
        accuracy = results['accuracy'] * 100
        print(f"   ✅ Accuracy: {accuracy:.2f}%")
        print(f"   ✅ Precision (Class 0): {results['detailed_report']['0']['precision']:.2f}\n")

if __name__ == "__main__":
    main()