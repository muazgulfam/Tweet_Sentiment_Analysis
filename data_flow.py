from graphviz import Digraph

dfd = Digraph(comment='Vertical Data Flow Diagram - Indo-Pak Twitter Sentiment Analysis')

# Layout settings for vertical readability
dfd.attr(rankdir='TB', size='10,20', dpi='300')
dfd.attr('node', fontname='Arial', fontsize='12')

# === External Entities ===
dfd.node('TwitterAPI', 'Twitter API', shape='box', style='filled', fillcolor='#f9ebae')
dfd.node('Researcher', 'Researcher', shape='box', style='filled', fillcolor='#c2f0c2')
dfd.node('User', 'General User', shape='box', style='filled', fillcolor='#c2f0c2')

# === Processes ===
dfd.node('P1', '1. Fetch & Preprocess Tweets', shape='ellipse', style='filled', fillcolor='#b3cde0')
dfd.node('P2', '2. Sentiment Analysis (VADER)', shape='ellipse', style='filled', fillcolor='#b3cde0')
dfd.node('P3', '3. Store in Supabase DB', shape='ellipse', style='filled', fillcolor='#b3cde0')
dfd.node('P4', '4. Generate Insights & Stats', shape='ellipse', style='filled', fillcolor='#b3cde0')
dfd.node('P5', '5. Visualize Data (Dashboard)', shape='ellipse', style='filled', fillcolor='#b3cde0')

# === Data Stores ===
dfd.node('DS1', 'Cleaned Tweet Dataset', shape='cylinder', style='filled', fillcolor='#fcd5ce')
dfd.node('DS2', 'Sentiment Results', shape='cylinder', style='filled', fillcolor='#fcd5ce')
dfd.node('DS3', 'Query Filters', shape='cylinder', style='filled', fillcolor='#fcd5ce')
dfd.node('DS4', 'Reports & Charts', shape='cylinder', style='filled', fillcolor='#fcd5ce')

# === Data Flows ===
dfd.edge('TwitterAPI', 'P1', label='Raw Tweets')
dfd.edge('P1', 'DS1', label='Cleaned Tweets')
dfd.edge('DS1', 'P2', label='Text Data')
dfd.edge('P2', 'DS2', label='Polarity Scores')
dfd.edge('P2', 'P3', label='Save to DB')
dfd.edge('P3', 'DS2', label='Stored Results')
dfd.edge('Researcher', 'P4', label='Generate Insights')
dfd.edge('DS2', 'P4', label='Sentiment Data')
dfd.edge('DS1', 'P4', label='Tweet Metadata')
dfd.edge('P4', 'DS4', label='Stats, Charts')
dfd.edge('User', 'P5', label='Search & Filter')
dfd.edge('DS2', 'P5', label='Filtered Sentiments')
dfd.edge('DS3', 'P5', label='Filter Rules')
dfd.edge('P5', 'User', label='Graphical Output')
dfd.edge('P4', 'Researcher', label='Insight Reports')

# === Render ===
dfd.render('vertical_data_flow_diagram', format='png', cleanup=True)
print("✅ Vertical DFD saved as 'vertical_data_flow_diagram.png'")
