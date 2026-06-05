# ==========================================
# IMPORT LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

div.stButton > button{
    background-color:#4CAF50;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}

div.stButton > button:hover{
    background-color:#45a049;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# FEATURES USED DURING TRAINING
# ==========================================
ALL_FEATURES = ["clean_review", "Review_Length"]


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("model_smv_sent6.pkl")

model = load_model()


# ==========================================
# TEXT CLEANING
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1+", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def review_word_count(text):
    return len(clean_text(text).split())


# ==========================================
# TITLE
# ==========================================
st.title("😊 Product Review Sentiment Analyzer")
st.write("Predict whether a customer review is Positive, Negative, or Neutral.")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.header("📝 Enter Review")

    review = st.text_area(
        "Customer Review",
        height=200,
        placeholder="Type your review here..."
    )

    st.divider()

    st.subheader("Examples")

    st.success("Positive\n\nThis product is amazing and arrived on time.")
    st.error("Negative\n\nTerrible quality. I want a refund.")
    st.info("Neutral\n\nThe package arrived yesterday.")


# ==========================================
# ANALYSIS
# ==========================================
if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        with st.spinner("Analyzing review..."):

            # CLEAN TEXT
            cleaned_review = clean_text(review)
            Review_Length = review_word_count(review)

            # CREATE INPUT DATAFRAME
            sentiment_df = pd.DataFrame({
                "clean_review": [cleaned_review],
                "Review_Length": [Review_Length]
            })

            sentiment_df = sentiment_df[ALL_FEATURES]


            # ==========================================
            # PREDICTION (FIXED LOGIC)
            # ==========================================
            proba = model.predict_proba(sentiment_df)[0]

            encoded_classes = model.classes_

            # Convert encoded labels to real text labels
            # THIS IS THE MAIN FIX
            try:
                label_encoder = joblib.load("label_encoder.pkl")
                classes = label_encoder.inverse_transform(encoded_classes)
            except:
                # fallback if no encoder available
                classes = encoded_classes

            best_index = proba.argmax()
            best_prob = proba[best_index]
            pred_label = classes[best_index]


        # ==========================================
        # RESULT
        # ==========================================
        st.header("Prediction Result")

        if pred_label == "Positive":
            st.success(f"😊 Predicted Sentiment: {pred_label}")

        elif pred_label == "Negative":
            st.error(f"😡 Predicted Sentiment: {pred_label}")

        else:
            st.warning(f"😐 Predicted Sentiment: {pred_label}")


        # ==========================================
        # METRICS
        # ==========================================
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence", f"{best_prob * 100:.2f}%")

        with col2:
            st.metric("Review Length", Review_Length)

        st.progress(float(best_prob))


        # ==========================================
        # TABS
        # ==========================================
        tab1, tab2, tab3 = st.tabs(
            ["Original Review", "Cleaned Text", "Probabilities"]
        )

        with tab1:
            st.info(review)

        with tab2:
            st.write(cleaned_review)

        with tab3:

            # Probability table
            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": proba
            })

            st.dataframe(probability_df, use_container_width=True)


            # ==========================================
            # CLEAN PROFESSIONAL BAR CHART
            # ==========================================
            fig, ax = plt.subplots(figsize=(8, 5))

            color_map = {
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#f39c12"
            }

            colors = [color_map.get(c, "#3498db") for c in classes]

            bars = ax.bar(classes, proba, color=colors, edgecolor="black")

            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Probability")
            ax.set_title("Prediction Confidence")

            ax.set_ylim(0, 1)

            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom"
                )

            st.pyplot(fig)


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-Learn + SVM")