import streamlit as st
import tweepy
import pandas as pd
import time
from textblob import TextBlob
import plotly.express as px

# --- Twitter API Setup ---
BEARER_TOKEN = st.secrets.get("BEARER_TOKEN", st.text_input("Enter Bearer Token", type="password"))

if not BEARER_TOKEN:
    st.warning("Please enter a valid Twitter Bearer Token")
    st.stop()

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# --- Improved Tweet Fetcher with Retries ---
def get_tweets_with_retry(query, max_tweets=30, max_retries=3):
    tweets = []
    retry_count = 0
    next_token = None
    
    while len(tweets) < max_tweets and retry_count < max_retries:
        try:
            # Get tweets with pagination
            response = client.search_recent_tweets(
                query=query,
                max_results=min(10, max_tweets - len(tweets)),  # Small batches
                next_token=next_token
            )
            
            if not response.data:
                break
                
            tweets.extend([{
                "Tweet": tweet.text,
                "Sentiment": analyze_sentiment(tweet.text),
                "War_Peace": detect_war_peace(tweet.text)
            } for tweet in response.data])
            
            # Check for more results
            next_token = response.meta.get('next_token')
            if not next_token:
                break
                
            # Small delay between batches
            time.sleep(2)
            
        except tweepy.errors.TooManyRequests:
            st.warning(f"Rate limited. Waiting 15 seconds (retry {retry_count + 1}/{max_retries})")
            time.sleep(15)
            retry_count += 1
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            break
    
    return pd.DataFrame(tweets[:max_tweets])  # Return up to max_tweets

# --- Analysis Functions ---
def analyze_sentiment(text):
    analysis = TextBlob(text)
    pol = analysis.sentiment.polarity
    if "pakistan" in text.lower() and "india" in text.lower():
        if pol > 0.1: return "Pro-Pakistan"
        elif pol < -0.1: return "Pro-India"
    return "Neutral"

def detect_war_peace(text):
    text_lower = text.lower()
    war_words = ["war", "attack", "military", "strike"]
    peace_words = ["peace", "dialogue", "talks", "negotiate"]
    war_count = sum(text_lower.count(word) for word in war_words)
    peace_count = sum(text_lower.count(word) for word in peace_words)
    return "Pro-War" if war_count > peace_count else "Pro-Peace" if peace_count > war_count else "Neutral"

# --- Streamlit App ---
st.title("🇮🇳🇵🇰 Real-time Sentiment Analysis")

query = st.text_input("Search Query", "(India Pakistan) lang:en -is:retweet")
tweet_count = st.slider("Tweet Count", 10, 50, 20)

if st.button("Analyze"):
    if not query:
        st.warning("Please enter a search query")
    else:
        with st.spinner(f"Fetching {tweet_count} tweets..."):
            df = get_tweets_with_retry(query, tweet_count)
            
        if not df.empty:
            st.success(f"Analyzed {len(df)} tweets")
            
            # Sentiment Pie Chart
            fig1 = px.pie(df, names='Sentiment', title='Sentiment Distribution')
            st.plotly_chart(fig1)
            
            # War/Peace Bar Chart
            fig2 = px.histogram(df, x='War_Peace', title='War vs Peace Stance')
            st.plotly_chart(fig2)
            
            # Show Sample Tweets
            st.subheader("Sample Tweets")
            st.dataframe(df.head())
        else:
            st.error("Failed to fetch tweets. Try reducing the count or waiting a few minutes.")