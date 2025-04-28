import streamlit as st
import requests
from newsapi import NewsApiClient
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------- Setup API keys --------------------
news_api_key = 'news_api'
openai_api_key = 'openai_api'

# Initialize clients
newsapi = NewsApiClient(api_key=news_api_key)
client = openai.OpenAI(api_key=openai_api_key)

# Download nltk resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# -------------------- Functions --------------------

def fetch_news(query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={news_api_key}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])
    return articles

def summarize_article(article_text, max_tokens=100):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following article:\n{article_text}"}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def analyze_sentiment(article_text):
    sentiment_score = sia.polarity_scores(article_text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_personalized_news(user_interest, sentiment_filter, summary_length):
    articles = fetch_news(user_interest)
    curated_content = []
    
    for article in articles:
        article_text = article.get('description') or article.get('content')
        if article_text:
            sentiment = analyze_sentiment(article_text)
            
            # Apply sentiment filter
            if sentiment_filter != 'Both' and sentiment != sentiment_filter:
                continue
            
            max_tokens = 50 if summary_length == 'Short' else 150
            summary = summarize_article(article_text, max_tokens=max_tokens)
            
            curated_content.append({
                'title': article['title'],
                'summary': summary,
                'sentiment': sentiment,
                'source': article['source']['name'],
                'url': article['url']
            })
    
    return curated_content

# -------------------- Streamlit UI --------------------

st.title("📰 NewsAgentAI - Personalized News Curator")

st.sidebar.header("Customize Your News")
user_interest = st.sidebar.text_input("Enter a Topic (e.g., AI, Space, Economy):", value="Artificial Intelligence")
sentiment_filter = st.sidebar.radio("Sentiment Filter:", ("Both", "Positive", "Negative"))
summary_length = st.sidebar.radio("Summary Length:", ("Short", "Detailed"))

if st.sidebar.button("Fetch News"):
    if user_interest:
        with st.spinner("Fetching and analyzing news..."):
            curated_news = generate_personalized_news(user_interest, sentiment_filter, summary_length)
        
        if curated_news:
            for news in curated_news:
                st.subheader(news['title'])
                st.write(f"**Summary:** {news['summary']}")
                st.write(f"**Sentiment:** {news['sentiment']}")
                st.write(f"**Source:** {news['source']}")
                st.markdown(f"[Read Full Article]({news['url']})")
                st.markdown("---")
        else:
            st.warning("No articles found matching your filters.")
    else:
        st.error("Please enter a topic to search.")

st.sidebar.markdown("---")

