from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"


def load_csv(filename, **kwargs):
    return pd.read_csv(DATA_DIR / filename, **kwargs)


def describe_dataframe(df, name):
    print(f"\n=== {name} ===")
    print(df.head())
    print(df.info(), "\n")


def save_plot(filename):
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def explore_education_vs_income():
    df = load_csv("education_vs_income_econometrics_samples.csv")
    describe_dataframe(df, "Education vs Income")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="x_education_years",
        y="y_income_thousands_usd",
        hue="sample",
        palette="deep",
    )
    plt.title("Education vs Income by Sample")
    plt.xlabel("Years of Education")
    plt.ylabel("Income (thousands USD)")
    save_plot("education_vs_income.png")
    return df


def explore_online_games():
    df = load_csv("online-gaming-10-04-26.csv", sep="\t")
    describe_dataframe(df, "Online Gaming")

    top_games = df.nlargest(10, "likes")[["name", "likes"]]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_games, y="name", x="likes", palette="viridis")
    plt.title("Top 10 Most Liked Online Games")
    plt.xlabel("Likes")
    plt.ylabel("")
    save_plot("top_online_games_by_likes.png")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["likes"], bins=30, kde=True)
    plt.title("Distribution of Likes")

    plt.subplot(1, 2, 2)
    sns.histplot(df["dislikes"], bins=30, kde=True, color="red")
    plt.title("Distribution of Dislikes")
    save_plot("online_game_reactions_distribution.png")
    return df


def explore_student_performance():
    df = load_csv("student_performance_finalscore.csv")
    describe_dataframe(df, "Student Performance")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Student Performance Features")
    save_plot("student_performance_correlation.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["Final_Score"], bins=30, kde=True)
    plt.title("Distribution of Final Exam Scores")
    plt.xlabel("Final Score")
    save_plot("final_score_distribution.png")
    return df


def explore_teen_mental_health():
    df = load_csv("Teen_Mental_Health_Dataset.csv")
    describe_dataframe(df, "Teen Mental Health")
    print(df["depression_label"].value_counts(), "\n")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="depression_label")
    plt.title("Depression Label Distribution")
    plt.xlabel("Depression Label (0 = No, 1 = Yes)")
    save_plot("depression_label_distribution.png")

    features = [
        "daily_social_media_hours",
        "sleep_hours",
        "screen_time_before_sleep",
        "academic_performance",
        "stress_level",
        "anxiety_level",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, feature in enumerate(features):
        row, col = divmod(i, 3)
        sns.boxplot(data=df, x="depression_label", y=feature, ax=axes[row][col])
        axes[row][col].set_title(f"{feature} vs Depression")
    save_plot("mental_health_feature_boxplots.png")
    return df


def explore_weather():
    df = load_csv("cities_weather_maharashtra.csv")
    describe_dataframe(df, "Maharashtra Weather")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cities_to_plot = ["Mumbai", "Pune", "Nagpur", "Nashik"]

    plt.figure(figsize=(12, 6))
    for city in cities_to_plot:
        city_data = df[df["city"] == city].dropna(subset=["tavg"])
        plt.plot(city_data["date"], city_data["tavg"], label=city)
    plt.title("Monthly Average Temperature Trends in Maharashtra Cities")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot("maharashtra_temperature_trends.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["prcp"].dropna(), bins=50, kde=True)
    plt.title("Precipitation Distribution")
    plt.xlabel("Precipitation (mm)")
    plt.xlim(0, 1000)
    save_plot("maharashtra_precipitation_distribution.png")
    return df


def create_preprocessor(categorical_cols, numeric_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def train_model(model, x_train, y_train, x_test, y_test, model_name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return accuracy


def build_ann(input_shape, hidden_units=(64, 32), dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(hidden_units[0], activation="relu", input_shape=(input_shape,)))
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units[1], activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def train_ann(model, x_train, y_train, x_test, y_test, model_name, epochs=30, batch_size=32):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    y_pred = (model.predict(x_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return history, accuracy


def plot_history(history, title_prefix, filename):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{title_prefix} Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"{title_prefix} Accuracy over Epochs")
    plt.legend()
    save_plot(filename)


def run_classification(df, epochs):
    target = "depression_label"
    x = df.drop(columns=[target])
    y = df[target]

    categorical_cols = ["gender", "platform_usage", "social_interaction_level"]
    numeric_cols = [
        "age",
        "daily_social_media_hours",
        "sleep_hours",
        "screen_time_before_sleep",
        "academic_performance",
        "physical_activity",
        "stress_level",
        "anxiety_level",
        "addiction_level",
    ]

    preprocessor = create_preprocessor(categorical_cols, numeric_cols)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = {}
    dt_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )
    results["Decision Tree"] = train_model(
        dt_model, x_train, y_train, x_test, y_test, "Decision Tree"
    )

    knn_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    results["KNN"] = train_model(
        knn_model, x_train, y_train, x_test, y_test, "K-Nearest Neighbors (k=5)"
    )

    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc = preprocessor.transform(x_test)

    ann_model = build_ann(x_train_proc.shape[1])
    history_ann, results["ANN"] = train_ann(
        ann_model,
        x_train_proc,
        y_train,
        x_test_proc,
        y_test,
        "ANN (64-32)",
        epochs=epochs,
    )
    plot_history(history_ann, "ANN", "ann_training_history.png")

    deep_model = build_ann(x_train_proc.shape[1], hidden_units=(128, 64), dropout_rate=0.3)
    history_deep, results["Deep ANN"] = train_ann(
        deep_model,
        x_train_proc,
        y_train,
        x_test_proc,
        y_test,
        "Deep ANN (128-64)",
        epochs=epochs,
    )
    plot_history(history_deep, "Deep ANN", "deep_ann_training_history.png")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis and classification models for the mini project."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for neural network models.",
    )
    args = parser.parse_args()

    explore_education_vs_income()
    explore_online_games()
    explore_student_performance()
    mental_health_df = explore_teen_mental_health()
    explore_weather()
    results = run_classification(mental_health_df, args.epochs)

    print("\n=== Model Accuracy Summary ===")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    print(f"\nSaved visualizations to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
