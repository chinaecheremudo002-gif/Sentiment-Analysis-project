# ---------------------------
# Web app framework
# ---------------------------
import streamlit as st    # Main library to make your app a website

# ---------------------------
# Data handling
# ---------------------------
import pandas as pd       
import numpy as np        

# ---------------------------
# Machine Learning
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE # for handling imbalanced datasets
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew, kurtosis
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_curve,
                             roc_auc_score,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Visualization
# ---------------------------
import matplotlib.pyplot as plt   # Basic charts
import seaborn as sns             # Pretty charts
import plotly.express as px       # Interactive charts


# to distinguish between categorical colum and numeric column

ALL_FEATURES=['clean_review', 'Review_Length']

@st.cache_resource # it tell stream_lit to load the model once and reuse them 

# Load trained model
def load_asset():
    model = joblib.load("model_smv_sent_vs.pkl")
    return model

model = load_asset()


# Load label encoder
def load_label():
    le = joblib.load("label_encoder.pkl")
    return le

label_encoder = load_label()
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# cleaning function
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"(.)\1+", r"\1", text) # fix repeated letters (sooo → so)
    text = re.sub(r"[^a-z\s]", "", text)  # remove numbers + symbols (noise)
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    words = text.split()  # tokenize
    return " ".join(words)

def review_word_count(text):
    return len(clean_text(text).split())
def main():
    st.title("Sentiment Prediction App")
    with st.sidebar:
        review=st.text_area("Enter your review",placeholder="Type your product review here..."
)     


    if st.button("Sentiment Analysis Prediction"):
        cleaned_review = clean_text(review)
        Review_Length=review_word_count(review)
        customer_input = {
                    "clean_review": [cleaned_review],
                    "Review_Length": [Review_Length]
}
        sentiment_df=pd.DataFrame( customer_input)  
        sentiment_df_copy=sentiment_df.copy()
        sentiment_df_features=sentiment_df_copy[ALL_FEATURES]
        
        pred_class = model.predict(sentiment_df_features)[0]

        proba = model.predict_proba(sentiment_df_features)[0]

        classes = model.classes_

        best_index = proba.argmax()
        best_class = classes[best_index]
        best_prob = proba[best_index]

        # Convert encoded prediction back to sentiment label
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        st.write("Sentiment Predicted:", pred_label)
        st.write("Best Class:", best_class)
        st.write(f"Confidence: {best_prob:.2f}")





if __name__ == "__main__":
    main()

