from graphviz import Digraph

dot = Digraph(comment='System Design Diagram')

# Main components
dot.node('A', 'Twitter API / Scraper')
dot.node('B', 'Data Preprocessing')
dot.node('C', 'Sentiment Analysis\n(VADER, TextBlob)')
dot.node('D', 'Emotion Classification\n(optional)')
dot.node('E', 'Database\n(Supabase)')
dot.node('F', 'Visualization Dashboard')
dot.node('G', 'User Interface')

# Connections
dot.edges(['AB', 'BC', 'CD', 'CE', 'EF'])
dot.edge('F', 'G', label='Interactive Charts')
dot.edge('A', 'E', label='Raw Tweets (Backup)', style='dashed')

# Optional
dot.attr(label='Mining Twitter for Indo-Pak Sentiment - System Design Diagram', labelloc='top', fontsize='20')

# Render diagram
dot.render('system_design_diagram', format='png', cleanup=True)
