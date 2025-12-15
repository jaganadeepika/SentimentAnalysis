import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

default_stopwords = set(stopwords.words("english"))
negation_words = {
    "not", "no", "nor", "n't", "don", "don't", "hadn't", "couldn't", "wouldn't",
    "won't", "wasn't", "aren't", "isn't", "haven't", "didn't", "doesn't", "weren't",
    "can't", "shouldn't", "never", "bad"
}
custom_stopwords = default_stopwords - negation_words
lemmatizer = WordNetLemmatizer()

neutral_words = {
    "amazon", "tablet", "kindle", "product", "device", "use", "used",
    "buy", "bought", "purchase", "purchased", "item", "got", "get",
    "one", "two", "day", "year", "time", "work", "working",
    "brand", "thing", "name", "price", "order", "received"
}
filtered_stopwords = custom_stopwords.union(neutral_words)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def remove_neutral_words(text):
    tokens = [w for w in text.split() if w not in filtered_stopwords]
    return " ".join(tokens)

@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        model = joblib.load("svm_model.pkl")
        return vectorizer, model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None


@st.cache_resource
def get_engine():
    user = "postgres"
    password = "moksha123"
    host = "localhost"
    db = "db_amazon_reviews"
    port = "5432"
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)

st.set_page_config(page_title="📊 Amazon Review Analytics", layout="wide")
st.title("📦 Amazon Negative Review Analytics Dashboard")

vectorizer, model = load_model()
engine = get_engine()

page = st.sidebar.radio("📂 Navigation", ["📥 Load Reviews", "📈 Analytics", "🔍 Predict Sentiment"])


if page == "📥 Load Reviews":
    st.header("📥 Load Negative Reviews from Database")
    query = "SELECT * FROM all_reviews WHERE sentiment = 'negative'"
    try:
        df = pd.read_sql(query, engine)
        st.success(f"Loaded {len(df)} negative reviews from database.")
        st.dataframe(df.head(10))
        if st.button("📥 Download as CSV"):
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download", csv, "negative_reviews.csv", "text/csv")
    except Exception as e:
        st.error(f"Failed to load data: {e}")

elif page == "📈 Analytics":
    st.header("📈 Review Analytics Dashboard")
    try:
      
        query_counts = """
        SELECT
            COUNT(*) AS total_reviews,
            SUM(CASE WHEN "reviews.rating" >= 3 THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN "reviews.rating" < 3 THEN 1 ELSE 0 END) AS negative_reviews
        FROM all_reviews
        """
        df_counts = pd.read_sql(query_counts, engine)
        total = df_counts.loc[0, 'total_reviews']
        positive = df_counts.loc[0, 'positive_reviews']
        negative = df_counts.loc[0, 'negative_reviews']

        st.metric("Total Reviews", total)
        st.metric("👍 Positive Reviews", positive)
        st.metric("👎 Negative Reviews", negative)

       
        st.subheader("📊 Sentiment Distribution")
        df_pie = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive, negative]
        })
        fig_pie = px.pie(df_pie, names="Sentiment", values="Count", color="Sentiment",
                         color_discrete_map={"Positive": "green", "Negative": "red"},
                         title="Overall Sentiment Distribution")
        st.plotly_chart(fig_pie)

      
        st.subheader("📆 Sentiment Trend Over Time")
        query_trend = """
        SELECT 
            substr("reviews.date", 1, 7) AS month,
            SUM(CASE WHEN "reviews.rating" >= 3 THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN "reviews.rating" < 3 THEN 1 ELSE 0 END) AS negative_reviews
        FROM all_reviews
        GROUP BY month
        ORDER BY month
        """
        df_trend = pd.read_sql(query_trend, engine)
        df_trend.set_index("month", inplace=True)
        st.line_chart(df_trend)

   
        st.subheader("🏷️ Top 10 Brands by Number of Complaints")
        query1 = """
        SELECT brand, COUNT(*) AS complaint_count
        FROM all_reviews
        WHERE "reviews.rating" < 3
        GROUP BY brand
        ORDER BY complaint_count DESC
        LIMIT 10
        """
        df_complaints = pd.read_sql(query1, engine)
        st.bar_chart(df_complaints.set_index('brand'))

        st.subheader("📦 Top 10 Products by Number of Complaints")
        query2 = """
        SELECT name, COUNT(*) AS complaints
        FROM all_reviews
        WHERE "reviews.rating" < 3
        GROUP BY name
        ORDER BY complaints DESC
        LIMIT 10
        """
        df_products = pd.read_sql(query2, engine)
        st.bar_chart(df_products.set_index('name'))

    
        st.subheader("📉 Complaints Over Time (Monthly)")
        query3 = """
        SELECT substr("reviews.date", 1, 7) AS month, COUNT(*) AS complaints
        FROM all_reviews
        WHERE "reviews.rating" < 3
        GROUP BY month
        ORDER BY month
        """
        df_time = pd.read_sql(query3, engine)
        st.line_chart(df_time.set_index("month"))


        st.subheader("☁️ Word Cloud of Negative Reviews")
        query4 = """
        SELECT "reviews.text"
        FROM all_reviews
        WHERE "reviews.rating" < 3 AND sentiment = 'negative' AND "reviews.text" IS NOT NULL
        """
        df_neg_texts = pd.read_sql(query4, engine)
        df_neg_texts['cleaned'] = df_neg_texts["reviews.text"].apply(clean_text)
        df_neg_texts['filtered'] = df_neg_texts['cleaned'].apply(remove_neutral_words)
        neg_text = " ".join(df_neg_texts['filtered'])
        wordcloud = WordCloud(width=800, height=400, background_color="white",
                              colormap="Reds", max_words=100).generate(neg_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        st.subheader("☁️ Word Cloud of Positive Reviews")
        query5 = """
        SELECT "reviews.text"
        FROM all_reviews
        WHERE "reviews.rating" >= 4 AND sentiment = 'positive' AND "reviews.text" IS NOT NULL
        """
        df_pos_texts = pd.read_sql(query5, engine)
        df_pos_texts['cleaned'] = df_pos_texts["reviews.text"].apply(clean_text)
        df_pos_texts['filtered'] = df_pos_texts['cleaned'].apply(remove_neutral_words)
        pos_text = " ".join(df_pos_texts['filtered'])
        wordcloud_pos = WordCloud(width=800, height=400, background_color="white",
                                  colormap="Greens", max_words=100).generate(pos_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud_pos, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


elif page == "🔍 Predict Sentiment":
    st.header("🔍 Sentiment Predictor")
    review_input = st.text_area("✍️ Enter a customer review:")

    if st.button("🔮 Predict"):
        if review_input.strip() == "":
            st.warning("Please enter some text.")
        elif model and vectorizer:
            cleaned = clean_text(review_input)
            X_input = vectorizer.transform([cleaned])
            prediction = model.predict(X_input)[0]
            label = "✅ Positive" if prediction == 1 else "❌ Negative"
            st.markdown(f"### Prediction: {label}")
        else:
            st.error("Model or vectorizer not loaded.")
