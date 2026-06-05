# ==========================================
# IMPORT LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import joblib
import re


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
# FEATURES
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
# SENTIMENT SCORE MAP
# ==========================================
score_map = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}


# ==========================================
# TITLE
# ==========================================
st.title("😊 Product Review Sentiment Analyzer")
st.write("AI-powered sentiment detection (SVM model - no probabilities)")


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

            # INPUT DATA
            sentiment_df = pd.DataFrame({
                "clean_review": [cleaned_review],
                "Review_Length": [Review_Length]
            })

            sentiment_df = sentiment_df[ALL_FEATURES]

            # PREDICTION (NO PROBABILITY)
            pred_class = model.predict(sentiment_df)[0]

            # Convert numeric label → text safely
            try:
                label_encoder = joblib.load("label_encoder.pkl")
                pred_label = label_encoder.inverse_transform([pred_class])[0]
            except:
                pred_label = pred_class


            # SENTIMENT SCORE
            sentiment_score = score_map.get(pred_label, 0)


        # ==========================================
        # RESULT
        # ==========================================
        st.header("Prediction Result")

        if pred_label == "Positive":
            st.success(f"😊 Sentiment: {pred_label}")

        elif pred_label == "Negative":
            st.error(f"😡 Sentiment: {pred_label}")

        else:
            st.warning(f"😐 Sentiment: {pred_label}")


        # ==========================================
        # METRICS
        # ==========================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sentiment Score", sentiment_score)

        with col2:
            st.metric("Review Length", Review_Length)

        with col3:
            st.metric("Word Count", Review_Length)


        # Progress bar (mapped)
        progress_value = (sentiment_score + 1) / 2  # convert -1,0,1 → 0-1
        st.progress(progress_value)


        # ==========================================
        # TABS
        # ==========================================
        tab1, tab2, tab3 = st.tabs(
            ["Original Review", "Cleaned Text", "Insights"]
        )

        with tab1:
            st.info(review)

        with tab2:
            st.write(cleaned_review)

        with tab3:

            # ==========================================
            # KEYWORD HIGHLIGHTING
            # ==========================================

            positive_words = ["good", "great", "amazing", "excellent", "love", "best"]
            negative_words = ["bad", "worst", "terrible", "hate", "poor", "awful"]

            words = cleaned_review.split()
            highlighted = []

            for w in words:
                if w in positive_words:
                    highlighted.append(f"🟢 {w}")
                elif w in negative_words:
                    highlighted.append(f"🔴 {w}")
                else:
                    highlighted.append(w)

            st.subheader("🔍 Keyword Insight")
            st.write(" ".join(highlighted))


            # ==========================================
            # MODEL EXPLANATION
            # ==========================================

            st.subheader("🧠 Why this prediction?")

            if "not" in cleaned_review:
                st.warning("Negation detected → may affect sentiment direction")

            if Review_Length < 3:
                st.info("Very short review → low confidence expected")

            if any(word in cleaned_review for word in positive_words):
                st.success("Positive words detected in text")

            if any(word in cleaned_review for word in negative_words):
                st.error("Negative words detected in text")


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-Learn + SVM (Explainable AI Version)")