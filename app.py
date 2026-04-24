import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("language.csv")  # keep CSV in same folder
    return df

df = load_data()

# -------------------------------
# Prepare Data
# -------------------------------
x = np.array(df["Text"])
y = np.array(df["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.60, random_state=99
)

# -------------------------------
# Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌍 Language Detection App")
st.write("Detect the language of any text using Machine Learning")

# Show dataset info
st.subheader("📊 Dataset Info")
st.write("Shape:", df.shape)
st.write("Languages:", df["language"].nunique())

# User input
user_input = st.text_area("✍️ Enter text:")

if st.button("Predict"):
    if user_input.strip() != "":
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)

        st.success(f"🌐 Predicted Language: {output[0]}")
    else:
        st.warning("Please enter some text!")

# Show accuracy
st.subheader("📈 Model Performance")
st.write(f"Accuracy: {accuracy:.2f}") 