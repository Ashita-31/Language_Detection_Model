import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Config + Blue Animated Gradient
# -------------------------------
st.set_page_config(page_title="Language Detector", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #4facfe, #00c6ff);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* MAIN BACKGROUND (your blue gradient) */
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #4facfe, #00c6ff);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
}

/* SIDEBAR LIGHTER GRADIENT */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4facfe, #00c6ff);
}

/* Optional: sidebar text color */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Animation */
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main text */
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("language.csv")

df = load_data()

# -------------------------------
# Prepare & Train Model
# -------------------------------
x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=99
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------------
# SIDEBAR (ALL CONTROLS HERE)
# -------------------------------
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider(
    "🎚️ Confidence Threshold",
    0.0, 1.0, 0.5, 0.01
)

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

# -------------------------------
# MAIN UI
# -------------------------------
st.title("🌍 Language Detection App")
st.write("Detect the language of any text using Machine Learning")

# Dataset info
st.subheader("📊 Dataset Info")
st.write("Shape:", df.shape)
st.write("Languages:", df["language"].nunique())

# Text input
user_input = st.text_area("✍️ Enter text:")

# -------------------------------
# SINGLE PREDICTION
# -------------------------------
if st.button("Predict"):
    if user_input.strip():
        with st.spinner("🔍 Detecting language..."):
            data = cv.transform([user_input])
            probs = model.predict_proba(data)

            confidence = np.max(probs)
            predicted_language = model.classes_[np.argmax(probs)]

        if confidence >= threshold:
            st.success(f"🌐 Language: {predicted_language}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.warning(f"⚠️ Low confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text!")

# -------------------------------
# CSV BATCH PREDICTION (SIDEBAR INPUT → MAIN OUTPUT)
# -------------------------------
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)

    if "Text" in df_upload.columns:
        with st.spinner("📂 Processing CSV..."):
            languages = []
            confidences = []

            for text in df_upload["Text"]:
                data = cv.transform([text])
                probs = model.predict_proba(data)

                confidence = np.max(probs)
                lang = model.classes_[np.argmax(probs)]

                languages.append(lang)
                confidences.append(confidence)

            df_upload["Predicted Language"] = languages
            df_upload["Confidence"] = confidences

        st.subheader("📄 CSV Results")
        st.write(df_upload)

        st.download_button(
            "⬇️ Download Results",
            df_upload.to_csv(index=False),
            "results.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must contain a 'Text' column")

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.subheader("📈 Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")