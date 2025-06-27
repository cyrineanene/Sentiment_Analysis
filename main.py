import pandas as pd
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from ClassificationModel.data_cleaning_training import DataPreprocessor
import dagshub
from dotenv import load_dotenv
# Load .env file with debugging
load_dotenv()

class TextModel:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = MultinomialNB()
        
    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.classifier.predict(X_test)

class Evaluation:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self, labels=None):
        self.f1 = f1_score(self.y_true, self.y_pred, average='binary', labels=labels, zero_division=0)
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        
    def print_metrics(self):
        print("F1 Score:", self.f1)
        print("Confusion Matrix:\n", self.confusion_matrix)
        print("Accuracy:", self.accuracy)
        return {"f1_score": self.f1, "accuracy": self.accuracy, "confusion_matrix": self.confusion_matrix.tolist()}

def get_processed_data(path):
    try:
        # Check dataset size
        df = pd.read_csv(path)
        dataset_size = len(df)
        print(f"Dataset loaded: {dataset_size} samples")
        if dataset_size < 50000:
            print(f"Dataset size ({dataset_size}) is less than 50,000. Skipping retraining.")
            return None, None, None
        # print("Raw dataset head:\n", df.head())
        # print("Raw label distribution in dataset:")
        # print(df['sentiment'].value_counts())
        # print("Unique raw sentiment values:", df['sentiment'].unique())

        # Encode sentiment labels ('positive'/'negative' to 1/0)
        label_encoder = LabelEncoder()
        df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
        # print("Encoded labels mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
        # print("Label distribution after encoding:")
        # print(df['sentiment_encoded'].value_counts())

        # Data preparation
        preprocessor = DataPreprocessor()
        corpus = preprocessor.preprocess(df)
        print(f"Corpus preprocessed: {len(corpus)} samples")
        return df, corpus, label_encoder
    except Exception as e:
        print(f"Error in get_processed_data: {e}")
        return None, None, None

def model_train(path):
    try:
        df, corpus, label_encoder = get_processed_data(path)
        
        # Model NLP fitting
        text_model = TextModel()
        X = text_model.vectorizer.fit_transform(corpus).toarray()
        y = df['sentiment_encoded']
        # print(f"Vectorized data: {X.shape}")
        # print("Label distribution for training:")
        # print(pd.Series(y).value_counts())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)
        # print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        # print("Train label distribution:")
        # print(pd.Series(y_train).value_counts())
        # print("Test label distribution:")
        # print(pd.Series(y_test).value_counts())

        # Train model
        text_model.train(X_train, y_train)
        return X_train, X_test, y_train, y_test, text_model, label_encoder
    except Exception as e:
        print(f"Error in model_train: {e}")
        return None, None, None, None, None, None

def model_predict(X_test, loaded_model, label_encoder):
    try:
        # print(f"Predicting on test data: {X_test.shape}")
        y_pred = loaded_model.predict(X_test)
        # print(f"Predictions made: {y_pred.shape}")
        # print("Prediction label distribution:")
        # print(pd.Series(y_pred).value_counts())
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        return y_pred, y_pred_labels
    except Exception as e:
        print(f"Error in model_predict: {e}")
        return None, None

def compare_and_promote_model(run_id, model_name, metrics, client):
    try:
        # Get current production model (if any)
        try:
            production_versions = client.get_latest_versions(model_name, stages=["Production"])
            current_f1 = 0.0
            if production_versions:
                for version in production_versions:
                    run = client.get_run(version.run_id)
                    current_f1 = run.data.metrics.get("f1_score", 0.0)
                    break
        except:
            current_f1 = 0.0
            print("No production model found, treating new model as best.")

        new_f1 = metrics["f1_score"]
        print(f"Comparing F1 scores: New={new_f1}, Current Production={current_f1}")
        if new_f1 > current_f1:
            print("New model has better F1 score. Promoting to Production.")
            model_uri = f"runs:/{run_id}/{model_name}"
            client.set_registered_model_tag(model_name, "best_model", "true")
            client.transition_model_version_stage(
                name=model_name,
                version=client.get_latest_versions(model_name, stages=["None"])[0].version,
                stage="Production",
                archive_existing_versions=True
            )
        else:
            print("New model does not outperform current production model. Keeping existing model.")
    except Exception as e:
        print(f"Error in compare_and_promote_model: {e}")

if __name__ == "__main__":
    
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

    # Initialize DagsHub and set MLflow tracking URI
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER", dagshub_username)
    dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME", "mlops-project")

    if not dagshub_username or not dagshub_token:
        raise ValueError("DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set as environment variables")

    try:
        # Initialize DagsHub with MLflow integration
        dagshub.init(
            repo_owner=dagshub_repo_owner,
            repo_name=dagshub_repo_name,
            mlflow=True
        )
        print(f"DagsHub initialized for repo: {dagshub_repo_owner}/{dagshub_repo_name}")
    except Exception as e:
        print(f"Error initializing DagsHub: {e}")
        exit(1)
    
    # Set MLflow tracking URI from environment variable
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

    # Set MLflow credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Set experiment name
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment_analysis")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Train model
    path = 'datasets/IMDB_Dataset.csv'
    X_train, X_test, y_train, y_test, text_model, label_encoder = model_train(path)
    print("Model training completed.")

    # Start MLflow run
    model_name = "MultinomialNB"
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("classifier", "MultinomialNB")

        # Predict and evaluate
        y_pred, y_pred_labels = model_predict(X_test, text_model, label_encoder)
        if y_pred is None:
            print("Prediction failed. Exiting.")
            exit(1)
        print("Prediction completed.")

        # Evaluate
        eval = Evaluation(y_test, y_pred)
        eval.calculate_metrics(labels=[0, 1])
        eval.print_metrics()
        print("Evaluation completed.")
        report_dict = {
            "confusion_matrix": eval.confusion_matrix.tolist(),
            "f1_score": eval.f1,
            "accuracy": eval.accuracy
        }
        # Log metrics
        mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'f1_score': report_dict['f1_score']
        })
        with open("confusion_matrix.txt", "w") as f:
                f.write(str(report_dict['confusion_matrix']))
        mlflow.log_artifact("confusion_matrix.txt")

        # Log and register model
        mlflow.sklearn.log_model(text_model.classifier, model_name)
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        # Compare and promote model
        client = mlflow.tracking.MlflowClient()
        compare_and_promote_model(run.info.run_id, model_name, report_dict, client)