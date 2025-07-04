import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
 
# Title and description
st.title("Regression model's Accuracy Checker")
st.write("Upload your dataset, select a target variable, and evaluate different regression models.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Select target variable
    target_variable = st.selectbox("Select the target variable (dependent variable):", data.columns)

    # Select features
    features = st.multiselect("Select feature columns (independent variables):", data.columns, default=[col for col in data.columns if col != target_variable])

    if target_variable and features:
        # Split data into train and test sets
        X = data[features]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        st.write("Choose regression models to evaluate:")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor (SVR)": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor()
        }
        selected_models = st.multiselect("Select models:", list(models.keys()), default=list(models.keys()))

        # Evaluate selected models
        results = {}
        for model_name in selected_models:
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {"MSE": mse, "R2 Score": r2}

        # Display results
        st.write("Model Evaluation Results:")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        # Highlight the best model based on R2 Score
        best_model = max(results, key=lambda x: results[x]["R2 Score"])
        st.success(f"The best model is **{best_model}** with an R2 Score of **{results[best_model]['R2 Score']:.4f}**.")
