import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from dash_table import DataTable
from flask_cors import CORS
from textwrap import shorten 
from flask import Flask, request  # Import Flask's request object

# Load CSV data
csv_path = r"Test Try 2.csv"
df = pd.read_csv(csv_path, encoding='latin1')

# Initialize the Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
CORS(app.server, resources={r"/": {"origins": ""}})

# CSS styles for improved alignment and layout
styles = {
    'container': {
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f0f0f0',
        'padding': '20px',
        'fontSize': '18px',
        'textAlign': 'center'
    },
    'dropdown_container': {
        'marginBottom': '20px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'width': '100%',
        'padding': '10px'
    },
    'dropdown': {
        'width': '48%',
        'padding': '8px',
        'fontSize': '18px'
    },
    'datatable_container': {
        'marginTop': '20px',
        'overflowX': 'scroll'
    },
    'datatable': {
        'border': '1px solid black',
        'fontSize': '16px'
    },
    'graph': {
        'height': 'calc(100vh - 100px)',
        'border': '1px solid black',
        'fontSize': '18px'
    }
}

# Function to limit text to 10 words with ellipsis
def limit_text(text, limit=10):
    if isinstance(text, str):
        if len(text.split()) > limit:
            return shorten(text, width=100, placeholder="...")
    return text

# Handle NaN values gracefully
df['Summary'] = df['Summary'].apply(limit_text)

# Select only the columns you want to display (feedback, fact, Feature, and Summary)
display_columns = ['fact', 'Feature', 'Summary', 'feedback']

# Define layout
app.layout = html.Div(style=styles['container'], children=[
    html.H3('Data Analysis Dashboard', style={'fontSize': '24px'}),
    html.Div(style=styles['dropdown_container'], children=[
        dcc.Dropdown(
            id='date-dropdown',
            options=[{'label': date, 'value': date} for date in df['date'].unique()],
            value=df['date'].unique()[0],
            style=styles['dropdown']
        ),
        dcc.Dropdown(
            id='brand-dropdown',
            options=[{'label': brand, 'value': brand} for brand in df['brand'].unique()],
            value=df['brand'].unique()[0],
            style=styles['dropdown']
        ),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': model, 'value': model} for model in df['model'].unique()],
            value=[df['model'].unique()[0]],
            multi=True,
            style=styles['dropdown']
        ),
    ]),
    html.Div(style=styles['datatable_container'], children=[
        DataTable(
            id='datatable',
            columns=[{'name': col, 'id': col} for col in display_columns],
            data=df[display_columns].to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'whiteSpace': 'normal'},
            page_size=20
        ),
    ]),
    html.Div(style=styles['graph'], children=[
        dcc.Graph(id='stacked-bar-graph')
    ]),
    html.Div(id='Summary-table', style={'marginTop': '20px', 'fontSize': '18px'})
])

# Callbacks remain the same
@app.callback(
    Output('datatable', 'data'),
    [Input('model-dropdown', 'value'), Input('date-dropdown', 'value'), Input('brand-dropdown', 'value')]
)
def update_datatable(selected_models, selected_date, selected_brand):
    if not selected_models:
        return []
    filtered_df = df[(df['model'].isin(selected_models)) & (df['date'] == selected_date) & (df['brand'] == selected_brand)]
    filtered_df['Summary'] = filtered_df['Summary'].apply(limit_text)
    return filtered_df[display_columns].to_dict('records')

@app.callback(
    Output('stacked-bar-graph', 'figure'),
    [Input('model-dropdown', 'value'), Input('date-dropdown', 'value'), Input('brand-dropdown', 'value')]
)
def update_stackedgraph(selected_models, selected_date, selected_brand):
    if not selected_models:
        return go.Figure()
    filtered_df = df[(df['model'].isin(selected_models)) & (df['date'] == selected_date) & (df['brand'] == selected_brand)]
    sentiment_counts = filtered_df['fact'].value_counts()
    total_count = sentiment_counts.sum()
    percentages = sentiment_counts / total_count * 100
    percentages = percentages.round(2)
    traces = []
    colors = ['#8b0000', '#d21401', '#545454', '#299617', '#234f1e']
    categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    for i, category in enumerate(categories):
        trace = go.Bar(
            x=[category for _ in selected_models],
            y=[percentages.get(category, 0) for _ in selected_models],
            name=category,
            marker_color=colors[i],
            hoverinfo='y+name',
            showlegend=i == 0
        )
        traces.append(trace)
    layout = go.Layout(
        barmode='stack',
        xaxis={'title': 'Sentiments', 'titlefont': {'size': 18}},
        yaxis={'title': 'Percentage', 'titlefont': {'size': 18}},
        legend={'tracegroupgap': 10},
        title={'text': '<b>Sentiment Analysis</b>', 'font': {'size': 22}}
    )
    return {'data': traces, 'layout': layout}


# Callback for the summary table
@app.callback(
    Output('Summary-table', 'children'),
    [Input('stacked-bar-graph', 'clickData'), Input('model-dropdown', 'value'), Input('date-dropdown', 'value'), Input('brand-dropdown', 'value')]
)
def update_Summary_table(clickData, selected_models, selected_date, selected_brand):
    if clickData is None:
        return html.Div("Click on a bar to see Summary.", style={'fontSize': '18px'})
    category = clickData['points'][0]['x']
    filtered_df = df[(df['fact'] == category) & (df['model'].isin(selected_models)) & (df['date'] == selected_date) & (df['brand'] == selected_brand)]
    Summary_data = filtered_df[['Summary']].head().to_dict('records')
    if not Summary_data:
        return html.Div(f"No Summary available for category: {category}", style={'fontSize': '18px'})

    # Construct the URL with query parameters
    url = f"/detailed_Summary?category={category}&models={','.join(selected_models)}&date={selected_date}&brand={selected_brand}"

    # Update layout with anchor tag
    return html.Div([
        html.H4(f'Summary for Category: {category}', style={'fontSize': '20px'}),
        html.A(f"View Detailed Summary", href=url, target="_blank", style={'fontSize': '18px'})
    ])

# Detailed Summary route
@app.server.route('/detailed_Summary')
def detailed_Summary():
    category = request.args.get('category')
    models = request.args.get('models').split(',')
    date = request.args.get('date')

    # Ensure we work with a copy to avoid SettingWithCopyWarning
    filtered_df = df[(df['fact'] == category) & (df['model'].isin(models)) & (df['date'] == date)].copy()

    # Select only the first 10 Summarys if available
    Summary_data = filtered_df.head()

    if Summary_data.empty:
        return f"<h1>No Summary available for {category}</h1>"

    # Create an HTML table
    table_html = '''<table border="1" style="font-size:20px; width:100%; border-collapse: collapse;">
                    <tr>
                        <th style="padding: 10px; text-align: left;">Brand</th>
                        <th style="padding: 10px; text-align: left;">Model</th>
                        <th style="padding: 10px; text-align: left;">Date</th>
                        <th style="padding: 10px; text-align: left;">Segment</th>
                        <th style="padding: 10px; text-align: left;">Summary</th>
                    </tr>'''

    for _, row in Summary_data.iterrows():
        table_html += f'''<tr>
                        <td style="padding: 10px;">{row['brand']}</td>
                        <td style="padding: 10px;">{row['model']}</td>
                        <td style="padding: 10px;">{row['date']}</td>
                        <td style="padding: 10px;">{row['segment']}</td>
                        <td style="padding: 10px;">{row['Summary']}</td>
                     </tr>'''

        table_html += '</table>'

        return f"<h1 style='font-size:24px;'>Detailed Summary for {category}</h1>{table_html}"

# Ensure this block is at the end
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8061))
    app.run_server(host='0.0.0.0', port=port, debug=False)
