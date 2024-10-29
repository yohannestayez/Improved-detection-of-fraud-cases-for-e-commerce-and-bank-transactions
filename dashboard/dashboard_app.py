from flask import Flask, jsonify, render_template
import pandas as pd
import requests
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)
dash_app = Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dashboard/',
    external_stylesheets=['/static/css/styles.css']
)

# Load the dataset
fraud_data = pd.read_csv('dashboard/data/merged_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/summary')
def data_summary():
    total_transactions = len(fraud_data)
    total_fraud = fraud_data['is_fraud'].sum()
    fraud_percentage = (total_fraud / total_transactions) * 100
    return jsonify({
        'total_transactions': total_transactions,
        'total_fraud': total_fraud,
        'fraud_percentage': fraud_percentage
    })

# Dashboard Layout
dash_app.layout = html.Div([
    html.H4("Fraud Detection Dashboard"),
    
    html.Div([
        html.Div(id='summary-stats', className='card')
    ], className="graph-container"),
    
    dcc.Graph(id='line-chart'),
    dcc.Graph(id='geo-map'),
    dcc.Graph(id='device-browser-bar-chart'),

    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

# Callbacks for Dashboard
@dash_app.callback(
    Output('summary-stats', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_summary_stats(n):
    response = requests.get('http://127.0.0.1:5000/data/summary').json()
    return html.Div([
        html.H4("Summary Statistics"),
        html.P(f"Total Transactions: {response['total_transactions']}"),
        html.P(f"Total Fraud Cases: {response['total_fraud']}"),
        html.P(f"Fraud Percentage: {response['fraud_percentage']:.2f}%")
    ])

@dash_app.callback(
    Output('line-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_line_chart(n):
    trends = fraud_data.groupby('purchase_time')['is_fraud'].sum().reset_index()
    fig = px.line(trends, x='purchase_time', y='is_fraud', title="Fraud Cases Over Time")
    return fig

# Add additional callbacks for geo-map and device-browser bar chart here

if __name__ == "__main__":
    app.run(debug=True)
