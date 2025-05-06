import tweepy
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Initialize NLTK
nltk.download('stopwords')

# Twitter API v2 credentials
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJgT1AEAAAAAM29z9nVGkCpgtMRdDq4DIE9PS4c%3D51re9A7rca8PIvWeGEbUiYuciy81HAG4mHv18xT8qVjYmQo1hc"
CLIENT_ID = "KhS4Qz98uHiS4VSelyEQqYuFd"
CLIENT_SECRET = "U8XkrNXz68RzWeMKZIe54VSUn4ijxonAUooc92LmF5wlkYSAko"

# Authenticate with Twitter API v2
try:
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=CLIENT_ID,
        consumer_secret=CLIENT_SECRET,
        wait_on_rate_limit=True
    )
    print("✅ Twitter API v2 Connected!")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    exit()

# Enhanced tweet fetching with war/peace detection
def get_tweets_v2(query, count=100):
    tweets = []
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=min(count, 100),
            tweet_fields=['created_at', 'author_id'],
            expansions=['author_id'],
            user_fields=['username']
        )
        
        users_cache = {}
        if response.includes and 'users' in response.includes:
            users_cache = {user.id: user.username for user in response.includes['users']}
        
        if response.data:
            for tweet in response.data:
                clean_text = clean_tweet(tweet.text)
                sentiment = analyze_sentiment(clean_text)
                war_peace = detect_war_peace(clean_text)
                
                tweets.append({
                    'Date': tweet.created_at,
                    'Tweet': tweet.text,
                    'Clean_Tweet': clean_text,
                    'Username': users_cache.get(tweet.author_id, str(tweet.author_id)),
                    'Sentiment': sentiment,
                    'War_Peace': war_peace
                })
                
                # Display tweet with analysis
                print(f"\n📢 Tweet: {tweet.text}")
                print(f"🧹 Cleaned: {clean_text}")
                print(f"🎭 Sentiment: {sentiment}")
                print(f"⚔️/🕊️ Stance: {war_peace}")
                print("-"*50)
                
        return pd.DataFrame(tweets)
    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return pd.DataFrame()

# Text cleaning function
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet.lower().strip()

# Enhanced sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if 'pakistan' in text and 'india' in text:
        if analysis.sentiment.polarity > 0.2:
            return 'Strongly Pro-Pakistan'
        elif analysis.sentiment.polarity > 0.1:
            return 'Pro-Pakistan'
        elif analysis.sentiment.polarity < -0.2:
            return 'Strongly Pro-India'
        elif analysis.sentiment.polarity < -0.1:
            return 'Pro-India'
    return 'Neutral'

# War/Peace stance detection
def detect_war_peace(text):
    war_terms = ['war', 'attack', 'military', 'strike', 'retaliate', 'conflict']
    peace_terms = ['peace', 'dialogue', 'talks', 'negotiate', 'ceasefire', 'diplomacy']
    
    war_count = sum(text.count(term) for term in war_terms)
    peace_count = sum(text.count(term) for term in peace_terms)
    
    if war_count > peace_count:
        return 'Pro-War'
    elif peace_count > war_count:
        return 'Pro-Peace'
    return 'Neutral'

# Search query
query = "(India Pakistan) OR (#IndiaPakistanConflict) OR (#PakIndoTensions) lang:en -is:retweet"

# Get and analyze tweets
print("⏳ Fetching and analyzing tweets...")
df = get_tweets_v2(query, 50)  # Reduced count for better analysis
if df.empty:
    print("⚠️ No tweets fetched. Exiting.")
    exit()

print(f"\n✅ Analysis Complete! Fetched {len(df)} tweets")

# Statistical Analysis
def print_statistics(df):
    # Sentiment Distribution
    print("\n📊 SENTIMENT DISTRIBUTION")
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)
    
    # War/Peace Stance
    print("\n⚔️🕊️ WAR vs PEACE STANCE")
    stance_counts = df['War_Peace'].value_counts()
    print(stance_counts)
    
    # Cross-analysis
    print("\n🔍 SENTIMENT BY STANCE")
    print(pd.crosstab(df['Sentiment'], df['War_Peace']))
    
    # Most common words
    all_words = ' '.join(df['Clean_Tweet']).split()
    word_counts = Counter(all_words)
    print("\n🗣️ TOP 10 WORDS:")
    print(word_counts.most_common(10))

# Visualizations
def create_visualizations(df):
    plt.figure(figsize=(15, 10))
    
    # Sentiment Distribution
    plt.subplot(2, 2, 1)
    df['Sentiment'].value_counts().plot(kind='bar', color=['green', 'lightgreen', 'gray', 'orange', 'red'])
    plt.title('Sentiment Analysis')
    plt.xticks(rotation=45)
    
    # War/Peace Stance
    plt.subplot(2, 2, 2)
    df['War_Peace'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'green', 'gray'])
    plt.title('Public Stance: War vs Peace')
    
    # Word Clouds
    plt.subplot(2, 2, 3)
    pro_war = ' '.join(df[df['War_Peace']=='Pro-War']['Clean_Tweet'])
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate(pro_war)
    plt.imshow(wordcloud)
    plt.title('Pro-War Keywords')
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    pro_peace = ' '.join(df[df['War_Peace']=='Pro-Peace']['Clean_Tweet'])
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate(pro_peace)
    plt.imshow(wordcloud)
    plt.title('Pro-Peace Keywords')
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# Generate reports
print_statistics(df)
create_visualizations(df)

# Save results
df.to_csv('tweet_analysis_results.csv', index=False)
print("\n💾 Results saved to 'tweet_analysis_results.csv'")