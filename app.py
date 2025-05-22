import requests
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from supabase import create_client, Client
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import time
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import LabelEncoder

# ========== Setup ==========

# Download NLTK stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords

# Supabase setup
SUPABASE_URL = "https://etcqfluqzeqxqiwcjdxs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV0Y3FmbHVxemVxeHFpd2NqZHhzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDcyMjYyNzIsImV4cCI6MjA2MjgwMjI3Mn0.EuDeCqCTdoE3LW8mPwCJjpePCBS9265o1-1N9CHNHl4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Twitter API setup
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAACmn1wEAAAAAvg03UqwV%2BAxkNhyqUZuXwONVFSc%3D4lXAWWTNGhCWudguOItl5k4oxDob3MByQdKJYqfcv8rUgisYGf"

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ========== Helper Functions ==========

def get_tweets_df():
    try:
        response = supabase.table('tweets').select('*').execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching tweets: {e}")
        return pd.DataFrame()

def get_sentiment_stats_df():
    try:
        response = supabase.table('sentiment_stats').select('*').execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching stats: {e}")
        return pd.DataFrame()

def fetch_tweets(query, max_results=50):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,author_id,lang,public_metrics,text"
    }
    url = "https://api.twitter.com/2/tweets/search/recent"
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        tweets = response.json().get("data", [])
        return pd.DataFrame(tweets)
    except Exception as e:
        print(f"❌ Twitter API error: {e}")
        return pd.DataFrame()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Pro-India'
    elif score <= -0.05:
        return 'Pro-Pakistan'
    else:
        return 'Neutral'

def eda(df):
    df['length'] = df['text'].apply(len)
    plt.figure(figsize=(8, 4))
    sns.histplot(df['length'], kde=True)
    plt.title("Tweet Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.savefig("tweet_length_dist.png")
    plt.close()

def store_data(df):
    for _, row in df.iterrows():
        try:
            supabase.table('tweets').insert({
                'text': row['text'],
                'created_at': row['created_at'].isoformat(),
                'sentiment': row['sentiment'],
                'region': row.get('region'),
                'hashtags': row.get('hashtags'),
                'geo': row.get('geo')
            }).execute()
            time.sleep(0.2)  # avoid rate limiting
        except Exception as e:
            print(f"Insert error: {e}")

def store_stats(df):
    stats = df['sentiment'].value_counts().reset_index()
    stats.columns = ['sentiment', 'count']
    stats['timestamp'] = datetime.now().isoformat()
    for _, row in stats.iterrows():
        try:
            supabase.table('sentiment_stats').insert({
                'sentiment': row['sentiment'],
                'count': int(row['count']),
                'timestamp': row['timestamp']
            }).execute()
        except Exception as e:
            print(f"Stats insert error: {e}")

def sentiment_by_hashtags(df):
    if 'hashtags' not in df.columns:
        return pd.DataFrame()
    hashtag_series = df['hashtags'].dropna().apply(lambda x: x.split(' ') if isinstance(x, str) else []).explode()
    hashtag_df = df[['hashtags', 'sentiment']].dropna().copy()
    hashtag_df['hashtag'] = hashtag_df['hashtags'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) else None)
    return hashtag_df.groupby(['hashtag', 'sentiment']).size().unstack().fillna(0)

def sentiment_misinformation_vs_factual(df):
    misinformation_keywords = ["fake", "hoax", "rumor"]
    factual_keywords = ["fact", "truth", "news"]
    df['misinformation'] = df['text'].apply(lambda x: any(word in x.lower() for word in misinformation_keywords))
    df['factual'] = df['text'].apply(lambda x: any(word in x.lower() for word in factual_keywords))
    return df.groupby(['misinformation', 'factual', 'sentiment']).size().unstack().fillna(0)

# ========== Statistical Analysis Functions ==========

def perform_chi2_test(data):
    try:
        chi2, p, dof, expected = chi2_contingency(data)
        return chi2, p
    except:
        return None, None

def perform_ttest(group1, group2):
    try:
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_val
    except:
        return None, None

def perform_ztest(count1, nobs1, count2, nobs2):
    try:
        count = [count1, count2]
        nobs = [nobs1, nobs2]
        z_stat, p_val = proportions_ztest(count, nobs)
        return z_stat, p_val
    except:
        return None, None

# ========== Enhanced Streamlit Dashboard ==========

def streamlit_dashboard():
    st.set_page_config(layout="wide", page_title="Mining Twitter for Indo-Pak Sentiment", page_icon="📊")
    # Main Title
    st.title("📊 Mining Twitter for Indo-Pak Sentiment")

    # Tagline
    st.markdown(
        "<h5 style='color: gray;'>Voices of the Subcontinent leverages an NLP and VADER-based approach "
        "to track sentiment trends in digital diplomacy.</h5>",
        unsafe_allow_html=True
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .tweet-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .pro-india { border-left: 5px solid #FF4B4B; }
    .pro-pakistan { border-left: 5px solid #4BFF4B; }
    .neutral { border-left: 5px solid #4B4BFF; }
    .main {background-color: #f5f5f5;}
    .stButton>button {border-radius: 5px;}
    .plot-container {background-color: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .section-header {color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;}
    .metric-card {background-color: white; border-radius: 10px; padding: 15px; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

    # Main title and description
    #st.title("🇵🇰🇮🇳 Comprehensive Pak-Indo Tweet Sentiment Analysis")
    st.markdown("""
    This dashboard provides a detailed statistical analysis of sentiment trends in Pakistan-India relations 
    based on Twitter data. All visualizations include statistical significance testing.
    """)

    # Load data
    df = get_tweets_df()
    if df.empty:
        st.warning("⚠️ No tweets found in database.")
        return
    
    # Convert date and add time-based features
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    df['polarity'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    # ========== 1. Top Tweets Display ==========
    st.markdown("## 📌 Top Tweets from Each Sentiment Category")
    
    # Get top 3-4 tweets from each sentiment category
    pro_india_tweets = df[df['sentiment'] == 'Pro-India'].sample(min(4, len(df[df['sentiment'] == 'Pro-India'])))
    pro_pakistan_tweets = df[df['sentiment'] == 'Pro-Pakistan'].sample(min(4, len(df[df['sentiment'] == 'Pro-Pakistan'])))
    neutral_tweets = df[df['sentiment'] == 'Neutral'].sample(min(4, len(df[df['sentiment'] == 'Neutral'])))

    # Display tweets in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Pro-India Tweets")
        for _, tweet in pro_india_tweets.iterrows():
            st.markdown(f"""
            <div class="tweet-card pro-india">
                <p>{tweet['text']}</p>
                <small>Posted at: {tweet['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Neutral Tweets")
        for _, tweet in neutral_tweets.iterrows():
            st.markdown(f"""
            <div class="tweet-card neutral">
                <p>{tweet['text']}</p>
                <small>Posted at: {tweet['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Pro-Pakistan Tweets")
        for _, tweet in pro_pakistan_tweets.iterrows():
            st.markdown(f"""
            <div class="tweet-card pro-pakistan">
                <p>{tweet['text']}</p>
                <small>Posted at: {tweet['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)

    # ========== 2. Overview Section ==========
    st.markdown("## 📊 Overview Dashboard", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Sentiment Distribution
        st.markdown("### Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        fig1 = px.pie(sentiment_counts, values='count', names='sentiment', 
                     color_discrete_sequence=['#FF4B4B', '#4BFF4B', '#4B4BFF'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Sentiment Over Time
        st.markdown("### Sentiment Over Time")
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
        fig2 = px.line(daily_sentiment, title="Daily Sentiment Trend",
                      color_discrete_map={
                          'Pro-India': '#FF4B4B',
                          'Pro-Pakistan': '#4BFF4B',
                          'Neutral': '#4B4BFF'
                      })
        st.plotly_chart(fig2, use_container_width=True)

    # ========== 3. War vs Peace Stance ==========
    st.markdown("## ⚔️ vs 🕊️ War vs Peace Stance Analysis")
    
    # Define keywords for war and peace
    war_keywords = ["war", "attack", "military", "strike", "conflict", "violence", "retaliate"]
    peace_keywords = ["peace", "dialogue", "talks", "negotiation", "ceasefire", "harmony", "cooperation"]
    
    df['war_peace'] = df['text'].apply(
        lambda x: "War" if any(word.lower() in x.lower() for word in war_keywords) else (
            "Peace" if any(word.lower() in x.lower() for word in peace_keywords) else "Neutral"
        )
    )
    
    war_peace_data = df.groupby('war_peace')['sentiment'].value_counts().unstack().fillna(0)
    
    if not war_peace_data.empty:
        fig = px.bar(war_peace_data, barmode='group',
                    title="Sentiment in War vs Peace Related Tweets",
                    color_discrete_map={
                        'Pro-India': '#FF4B4B',
                        'Pro-Pakistan': '#4BFF4B',
                        'Neutral': '#4B4BFF'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical test
        chi2, p = perform_chi2_test(war_peace_data)
        if chi2 is not None:
            st.markdown("""
            **Statistical Significance (Chi-Square Test):**
            - χ² = {:.2f}, p-value = {:.4f}
            - {}
            """.format(chi2, p, 
                      "✅ Statistically significant differences exist" if p < 0.05 else 
                      "❌ No significant differences found"))

    # ========== 4. Issue-Specific Comparisons ==========
    st.markdown("## 🎯 Issue-Specific Sentiment Analysis")
    
    # Define issue-specific keywords
    issues = {
        "Kashmir-Specific Sentiment": ["kashmir", "kashmiri", "jammu", "article 370"],
        "Cross-Border Terrorism vs National Sovereignty": ["terrorism", "terrorist", "sovereignty", "border"],
        "Ceasefire Violations: Who is Responsible?": ["ceasefire", "violation", "loc", "border firing"],
        "Economic Relations": ["trade", "economy", "business", "export", "import"],
        "Water Disputes": ["water", "indus", "treaty", "river"]
    }
    
    for issue_name, keywords in issues.items():
        st.markdown(f"### {issue_name}")
        
        # Filter tweets containing these keywords
        issue_tweets = df[df['text'].str.contains('|'.join(keywords), case=False)]
        
        if not issue_tweets.empty:
            # Sentiment distribution
            issue_data = issue_tweets['sentiment'].value_counts(normalize=True).reset_index()
            issue_data.columns = ['sentiment', 'proportion']
            
            fig = px.pie(issue_data, values='proportion', names='sentiment',
                        title=f"Sentiment Distribution for {issue_name}",
                        color='sentiment',
                        color_discrete_map={
                            'Pro-India': '#FF4B4B',
                            'Pro-Pakistan': '#4BFF4B',
                            'Neutral': '#4B4BFF'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with overall sentiment
            overall_sentiment = df['sentiment'].value_counts(normalize=True)
            comparison_df = pd.DataFrame({
                'Overall': overall_sentiment,
                issue_name: issue_tweets['sentiment'].value_counts(normalize=True)
            }).fillna(0)
            
            fig_compare = px.bar(comparison_df, barmode='group',
                                title=f"Comparison with Overall Sentiment",
                                color_discrete_map={
                                    'Pro-India': '#FF4B4B',
                                    'Pro-Pakistan': '#4BFF4B',
                                    'Neutral': '#4B4BFF'
                                })
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Statistical comparison for each sentiment
            st.markdown("**Statistical Significance (Z-Test for Proportions):**")
            for sentiment in ['Pro-India', 'Pro-Pakistan', 'Neutral']:
                try:
                    issue_count = (issue_tweets['sentiment'] == sentiment).sum()
                    issue_total = len(issue_tweets)
                    overall_count = (df['sentiment'] == sentiment).sum()
                    overall_total = len(df)
                    
                    z_stat, p_val = perform_ztest(issue_count, issue_total, overall_count, overall_total)
                    if z_stat is not None:
                        st.write(f"- {sentiment}: z-score = {z_stat:.2f}, p-value = {p_val:.4f} ({'✅ Significant' if p_val < 0.05 else '❌ Not significant'})")
                except:
                    pass
        else:
            st.warning(f"No tweets found for {issue_name}")

    # ========== 5. Diplomatic Relations Analysis ==========
    st.markdown("## 🏛️ Diplomatic Relations Sentiment")
    
    diplomatic_keywords = ["diplomat", "embassy", "high commission", "foreign minister", "foreign secretary"]
    diplomatic_tweets = df[df['text'].str.contains('|'.join(diplomatic_keywords), case=False)]
    
    if not diplomatic_tweets.empty:
        # Sentiment over time for diplomatic tweets
        diplomatic_tweets['month'] = diplomatic_tweets['created_at'].dt.to_period('M').astype(str)
        monthly_diplomatic = diplomatic_tweets.groupby(['month', 'sentiment']).size().unstack().fillna(0)
        
        fig = px.line(monthly_diplomatic, 
                     title="Monthly Sentiment in Diplomatic Relations Tweets",
                     color_discrete_map={
                         'Pro-India': '#FF4B4B',
                         'Pro-Pakistan': '#4BFF4B',
                         'Neutral': '#4B4BFF'
                     })
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare with overall sentiment
        diplomatic_sentiment = diplomatic_tweets['sentiment'].value_counts(normalize=True)
        comparison_df = pd.DataFrame({
            'Overall': df['sentiment'].value_counts(normalize=True),
            'Diplomatic': diplomatic_sentiment
        }).fillna(0)
        
        fig_compare = px.bar(comparison_df, barmode='group',
                            title="Diplomatic vs Overall Sentiment",
                            color_discrete_map={
                                'Pro-India': '#FF4B4B',
                                'Pro-Pakistan': '#4BFF4B',
                                'Neutral': '#4B4BFF'
                            })
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.warning("No diplomatic relations tweets found")

    # ========== 6. Hashtag Analysis ==========
    

    # ========== 7. Media vs Public Sentiment ==========
    st.markdown("## 📰 News Media vs Public Sentiment")
    st.info("Note: Media accounts are simulated for demonstration")
    
    # Simulate media accounts
    media_accounts = ['BBCIndia', 'PTI_News', 'Dawn_News', 'TheHindu']
    df['is_media'] = df['text'].str.contains('|'.join(media_accounts), case=False)
    
    if df['is_media'].any():
        media_data = df.groupby('is_media')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
        media_data.index = media_data.index.map({True: 'Media', False: 'Public'})
        
        fig = px.bar(media_data, barmode='group',
                    title="Media vs Public Sentiment Comparison",
                    color_discrete_sequence=['#FF4B4B', '#4BFF4B', '#4B4BFF'])
        st.plotly_chart(fig, use_container_width=True)
        
        # T-test for sentiment polarity
        st.markdown("### Statistical Comparison")
        media_scores = df[df['is_media']]['polarity']
        public_scores = df[~df['is_media']]['polarity']
        
        t_stat, p_val = perform_ttest(media_scores, public_scores)
        if t_stat is not None:
            st.markdown("""
            **Independent Samples T-Test:**
            - t-statistic = {:.2f}, p-value = {:.4f}
            - {}
            """.format(t_stat, p_val,
                      "✅ Significant difference detected" if p_val < 0.05 else 
                      "❌ No significant difference found"))
    else:
        st.warning("No media tweets found in dataset")

    # ========== 8. Misinformation Analysis ==========
    

    # ========== 9. Sports Events Analysis ==========
    st.markdown("## 🏏 Sentiment Around India vs Pakistan Sports Matches")
    st.info("Note: Match dates are simulated for demonstration")
    
    # Simulated match dates
    matches = {
        "Cricket World Cup 2023": "2023-10-14",
        "Asia Cup 2023": "2023-09-02",
        "T20 World Cup 2022": "2022-10-23"
    }
    
    for match_name, match_date in matches.items():
        st.markdown(f"### {match_name}")
        match_date = pd.to_datetime(match_date)
        
        # Filter data around match date
        match_data = df[(df['created_at'] >= match_date - timedelta(days=2)) & 
                       (df['created_at'] <= match_date + timedelta(days=2))]
        
        if not match_data.empty:
            # Sentiment over time
            match_data['hour'] = match_data['created_at'].dt.floor('H')
            hourly_sentiment = match_data.groupby(['hour', 'sentiment']).size().unstack().fillna(0)
            
            fig = px.line(hourly_sentiment, title=f"Hourly Sentiment Around {match_name}",
                         color_discrete_map={
                             'Pro-India': '#FF4B4B',
                             'Pro-Pakistan': '#4BFF4B',
                             'Neutral': '#4B4BFF'
                         })
            fig.add_vline(x=match_date, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No sports events tweets found")

    # ========== 10. Polarity Distribution ==========
    st.markdown("## 📉 Sentiment Polarity Distribution")
    
    fig = px.histogram(df, x='polarity', nbins=50,
                      title="Distribution of Sentiment Polarity Scores",
                      color_discrete_sequence=['#636EFA'])
    fig.add_vline(x=0.05, line_dash="dash", line_color="green")
    fig.add_vline(x=-0.05, line_dash="dash", line_color="red")
    fig.update_layout(annotations=[
        dict(x=0.3, y=0.9, xref="paper", yref="paper",
             text="Pro-India Threshold", showarrow=False),
        dict(x=0.7, y=0.9, xref="paper", yref="paper",
             text="Pro-Pakistan Threshold", showarrow=False)
    ])
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical measures
    st.markdown("### Statistical Measures")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{df['polarity'].mean():.2f}")
    col2.metric("Std Dev", f"{df['polarity'].std():.2f}")
    col3.metric("Skewness", f"{df['polarity'].skew():.2f}")
    col4.metric("Kurtosis", f"{df['polarity'].kurtosis():.2f}")

    # ========== Data Download ==========
    st.markdown("---")
    st.markdown("## 📥 Download Data")
    if st.button("Download Current Analysis as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="pak_india_sentiment_analysis.csv",
            mime="text/csv"
        )

# ========== Main Orchestrator ==========

def main():
    query = "Pakistan India conflict -is:retweet lang:en"
    print("📥 Fetching tweets...")
    df = fetch_tweets(query)
    if df.empty:
        print("⚠️ No tweets found.")
        return

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['text'] = df['text'].fillna("")
    df['clean_text'] = df['text'].apply(preprocess)

    print("🔍 Analyzing Sentiment...")
    df['sentiment'] = df['clean_text'].apply(analyze_sentiment)

    print("📊 Running EDA...")
    eda(df)

    print("🗄️ Storing to Supabase...")
    store_data(df)
    store_stats(df)

    print("🚀 Launching Dashboard...")
    streamlit_dashboard()

if __name__ == "__main__":
    main()