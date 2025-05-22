from graphviz import Digraph

dot = Digraph(comment='Indo-Pak Tweet Sentiment Analysis - Activity Diagram', format='png')
dot.attr(rankdir='TB', size='8,10')
dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Helvetica')

# Start and End
dot.node('Start', 'Start', shape='circle', fillcolor='lightgreen')
dot.node('End', 'End', shape='circle', fillcolor='red')

# Activities
dot.node('DefineScope', 'Define Project Scope')
dot.node('ConnectAPI', 'Connect to Twitter API (Bearer Token)')
dot.node('ScrapeTweets', 'Scrape Tweets (Hashtags, Keywords)')
dot.node('StoreRaw', 'Store Raw Tweets in Supabase DB')
dot.node('Preprocessing', 'Preprocess Tweets\n(Cleaning, Lowercasing, Stopwords Removal)')
dot.node('EDA', 'Exploratory Data Analysis\n(Frequency, Wordcloud, Distribution)')
dot.node('Sentiment', 'Apply Sentiment Analysis\n(VADER, Polarity Score)')
dot.node('Categorize', 'Categorize by\nCountry / Region / Event')
dot.node('Visualize', 'Visualize Data\n(Bar, Pie, Timeline, Heatmaps)')
dot.node('Compare', 'Statistical Comparison\n(Pak vs India, Event-wise, Hashtag-wise)')
dot.node('StoreResults', 'Store Results in Supabase')
dot.node('GenerateReport', 'Generate Summary Report')
dot.node('Feedback', 'Gather Feedback for Improvements')

# Connections
dot.edges([
    ('Start', 'DefineScope'),
    ('DefineScope', 'ConnectAPI'),
    ('ConnectAPI', 'ScrapeTweets'),
    ('ScrapeTweets', 'StoreRaw'),
    ('StoreRaw', 'Preprocessing'),
    ('Preprocessing', 'EDA'),
    ('EDA', 'Sentiment'),
    ('Sentiment', 'Categorize'),
    ('Categorize', 'Visualize'),
    ('Visualize', 'Compare'),
    ('Compare', 'StoreResults'),
    ('StoreResults', 'GenerateReport'),
    ('GenerateReport', 'Feedback'),
    ('Feedback', 'End')
])

# Render
dot.render('activity_diagram', view=True)