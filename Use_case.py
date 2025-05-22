from graphviz import Digraph

dot = Digraph(comment='Use Case Diagram - Indo-Pak Twitter Sentiment Analysis')
dot.attr(rankdir='LR', size='8')

# Actors
dot.node('Researcher', shape='actor')
dot.node('Admin', shape='actor')
dot.node('User', 'General User', shape='actor')
dot.node('TwitterAPI', 'Twitter API', shape='actor')
dot.node('DB', 'Database', shape='actor')
dot.node('Sentiment', 'Sentiment Analyzer (VADER)', shape='actor')

# Use Cases
use_cases = {
    'FetchTweets': 'Fetch Tweets',
    'PreprocessData': 'Preprocess Data',
    'AnalyzeSentiment': 'Perform Sentiment Analysis',
    'StoreData': 'Store in Database',
    'ViewStats': 'View Statistical Comparison\n(India vs Pakistan)',
    'FilterData': 'Filter by Region / Hashtag',
    'EventAnalysis': 'Analyze Before/After Event',
    'ViewDashboard': 'Display Dashboard',
    'ExportReports': 'Export Reports',
    'ManageUsers': 'Manage User Roles',
    'ViewInsights': 'View Insights'
}

for key, label in use_cases.items():
    dot.node(key, label, shape='ellipse')

# Relationships - Researcher
dot.edge('Researcher', 'FetchTweets')
dot.edge('Researcher', 'PreprocessData')
dot.edge('Researcher', 'AnalyzeSentiment')
dot.edge('Researcher', 'ViewStats')
dot.edge('Researcher', 'FilterData')
dot.edge('Researcher', 'EventAnalysis')
dot.edge('Researcher', 'ExportReports')
dot.edge('Researcher', 'ViewInsights')

# Relationships - Admin
dot.edge('Admin', 'ManageUsers')
dot.edge('Admin', 'StoreData')

# Relationships - General User
dot.edge('User', 'ViewDashboard')
dot.edge('User', 'ViewStats')
dot.edge('User', 'FilterData')

# Twitter API and Sentiment Analyzer
dot.edge('TwitterAPI', 'FetchTweets')
dot.edge('Sentiment', 'AnalyzeSentiment')

# DB usage
dot.edge('StoreData', 'DB', label='writes to')
dot.edge('ViewStats', 'DB', label='reads from', style='dashed')
dot.edge('FilterData', 'DB', style='dashed')
dot.edge('EventAnalysis', 'DB', style='dashed')
dot.edge('ViewDashboard', 'DB', style='dashed')

# Render diagram
dot.render('use_case_diagram', format='png', cleanup=True)
print("Use case diagram generated successfully as 'use_case_diagram.png'")
