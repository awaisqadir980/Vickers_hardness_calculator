import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # Importing the experimental feature
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier  # Importing XGBoost
import pickle

# 1. Welcome message
st.title("Welcome to the ML App")
st.write("This application allows you to build and evaluate machine learning models with ease.")

# 2. Data upload or use example data
data_choice = st.sidebar.radio("Do you want to upload your data or use an example dataset?", ("Upload Data", "Use Example Data"))

if data_choice == "Upload Data":
    # 3. File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "tsv"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t')
else:
    # 4. Example dataset selection
    dataset_name = st.sidebar.selectbox("Select a dataset", ["titanic", "tips", "iris"])
    df = sns.load_dataset(dataset_name)

if 'df' in locals():
    # 5. Data information
    st.write("### Data Head")
    st.write(df.head())
    
    st.write("### Data Shape")
    st.write(df.shape)
    
    st.write("### Data Description")
    st.write(df.describe())
    
    st.write("### Data Info")
    buffer = st.empty()
    df.info(buf=buffer)
    st.text(buffer)

    st.write("### Column Names")
    st.write(df.columns)

    # 6. Feature and target selection
    features = st.multiselect("Select the features", df.columns)
    target = st.selectbox("Select the target", df.columns)
    
    if features and target:
        X = df[features]
        y = df[target]

        # 7. Problem identification
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 20:
                problem_type = 'regression'
                st.write("This is a regression problem.")
            else:
                problem_type = 'classification'
                st.write("This is a classification problem.")
        else:
            problem_type = 'classification'
            st.write("This is a classification problem.")

        # 8. Data Pre-processing
        if X.isnull().values.any() or y.isnull().values.any():
            imputer = IterativeImputer()
            X = pd.DataFrame(imputer.fit_transform(X), columns=features)
            y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())
            st.write("Missing values have been filled.")

        encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            encoders[column] = le
            st.write(f"Encoded {column} with LabelEncoder.")

        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            encoders['target'] = le
            st.write(f"Encoded target column {target} with LabelEncoder.")

        # 9. Train-test split
        test_size = st.slider("Select the test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 10. Model selection
        if problem_type == 'regression':
            model_choice = st.sidebar.selectbox("Select a model", ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "XGBoost"])  # Add XGBoost option
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_choice == "Random Forest":
                model = RandomForestRegressor()
            elif model_choice == "Support Vector Machine":
                model = SVR()
            elif model_choice == "XGBoost":  # Add XGBoost model
                model = XGBRegressor()
        else:
            model_choice = st.sidebar.selectbox("Select a model", ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "XGBoost"])  # Add XGBoost option
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            elif model_choice == "Support Vector Machine":
                model = SVC()
            elif model_choice == "XGBoost":  # Add XGBoost model
                model = XGBClassifier()

        # 11. Model training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 12. Model evaluation
        if problem_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"### {model_choice} Evaluation")
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Root Mean Squared Error: {rmse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R2 Score: {r2}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            st.write(f"### {model_choice} Evaluation")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")
            st.write("Confusion Matrix:")
            st.write(cm)

        # 13. Model evaluation
if problem_type == 'regression':
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### {model_choice} Evaluation")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"R2 Score: {r2}")
else:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    st.write(f"### {model_choice} Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")
    st.write("Confusion Matrix:")
    st.write(cm)

# 14. Model comparison
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write("### Actual vs Predicted Values")
st.write(comparison_df)

# 15. Highlight the best model (simplified for this example, can be expanded)
st.write(f"The {model_choice} model has been evaluated.")
