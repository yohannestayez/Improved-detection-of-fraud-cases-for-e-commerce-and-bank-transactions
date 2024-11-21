from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
import pandas as pd
from flask import Flask
from datetime import datetime
import socket
import struct

# Initialize Flask app
server = Flask(__name__)

# Initialize Dash app
app = Dash(__name__, server=server)

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return None  # Handle invalid IPs gracefully

# Load datasets
def load_data():
    fraud_data = pd.read_csv('data/Fraud_Data.csv')
    credit_data = pd.read_csv('data/render_creditcard.csv')
    ip_country = pd.read_csv('data/IpAddress_to_Country.csv')
    return fraud_data, credit_data, ip_country

# Data processing functions
def process_ecommerce_data(fraud_data, ip_country):
    fraud_data_cleaned = fraud_data.copy()
    ip_country_cleaned = ip_country.copy()

    fraud_data_cleaned['signup_time'] = pd.to_datetime(fraud_data_cleaned['signup_time'])
    fraud_data_cleaned['purchase_time'] = pd.to_datetime(fraud_data_cleaned['purchase_time'])
    fraud_data_cleaned['purchase_day'] = fraud_data_cleaned['purchase_time'].dt.day_name()
    fraud_data_cleaned['purchase_hour'] = fraud_data_cleaned['purchase_time'].dt.hour
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_address'].apply(lambda x: ip_to_int(str(int(x))) if pd.notna(x) else None)
    
    ip_country_cleaned['lower_bound_ip_address'] = ip_country_cleaned['lower_bound_ip_address'].astype('int64')
    ip_country_cleaned['upper_bound_ip_address'] = ip_country_cleaned['upper_bound_ip_address'].astype('int64')
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_int'].astype('int64')
    
    ip_country_cleaned.sort_values('lower_bound_ip_address', inplace=True)
    fraud_data_with_country = pd.merge_asof(
        fraud_data_cleaned.sort_values('ip_int'),
        ip_country_cleaned[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    fraud_data_with_country = fraud_data_with_country[
        (fraud_data_with_country['ip_int'] >= fraud_data_with_country['lower_bound_ip_address']) &
        (fraud_data_with_country['ip_int'] <= fraud_data_with_country['upper_bound_ip_address'])
    ]
    fraud_data_with_country.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    
    return fraud_data_with_country

def create_summary_stats(fraud_data, credit_data):
    ecom_stats = {
        'total_transactions': len(fraud_data),
        'fraud_cases': fraud_data['class'].sum(),
        'fraud_percentage': (fraud_data['class'].sum() / len(fraud_data) * 100).round(2)
    }
    credit_stats = {
        'total_transactions': len(credit_data),
        'fraud_cases': credit_data['Class'].sum(),
        'fraud_percentage': (credit_data['Class'].sum() / len(credit_data) * 100).round(2)
    }
    return ecom_stats, credit_stats

# Load and process data
fraud_data, credit_data, ip_country = load_data()
fraud_data_processed = process_ecommerce_data(fraud_data, ip_country)
ecom_stats, credit_stats = create_summary_stats(fraud_data_processed, credit_data)

# App layout
app.layout = html.Div([
    # Navigation bar
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),
        html.P('fraud analytics and insights', className='nav-subtitle')
    ], className='navbar'),

    # Filters
    html.Div([
        dcc.DatePickerRange(
            id='date-filter',
            start_date=fraud_data_processed['purchase_time'].min(),
            end_date=fraud_data_processed['purchase_time'].max(),
            display_format='YYYY-MM-DD',
            className='date-picker'
        ),
        html.Button('Filter', id='filter-button', className='filter-button'),
        html.Button('Reset Filters', id='reset-button', className='filter-button')
    ], className='filters-container'),

    # Alert message
    html.Div(id='alert-message', className='alert-message', style={'display': 'none'}),

    # Main content container
    html.Div([
        # Summary Statistics Cards
        html.Div([
            html.Div([
                html.Div([
                    html.I(className='fas fa-shopping-cart stat-icon'),
                    html.Div([
                        html.H3('E-commerce Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{ecom_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6'),

            html.Div([
                html.Div([
                    html.I(className='fas fa-credit-card stat-icon'),
                    html.Div([
                        html.H3('Credit Card Transactions'),
                        html.Div([
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{credit_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')
            ], className='col-md-6')
        ], className='row stats-container'),

        # Charts Section
        html.Div([
            html.Div([
                html.H3('Fraud Trends Over Time', className='chart-title'),
                dcc.Graph(id='fraud-trends')
            ], className='chart-card mb-4'),

            html.Div([
                html.H3('Geographical Distribution of Fraud', 
                        className='chart-title', 
                        style={'textAlign': 'center', 'marginBottom': '15px'}),
                dcc.Graph(id='geo-distribution', className='chart-card mb-4')
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'}),

            # Device and Browser Analysis
            html.Div([
                html.Div([
                    html.H3('Fraud by Device', className='chart-title'),
                    dcc.Graph(id='fraud-device')
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Browser', className='chart-title'),
                    dcc.Graph(id='fraud-browser')
                ], className='chart-card col-md-6')
            ], className='row mb-4'),

            # Time Patterns
            html.Div([
                html.Div([
                    html.H3('Fraud by Hour of Day', className='chart-title'),
                    dcc.Graph(id='fraud-hour')
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Day of Week', className='chart-title'),
                    dcc.Graph(id='fraud-day')
                ], className='chart-card col-md-6')
            ], className='row mb-4')
        ], className='charts-container')
    ], className='main-content')
])

# Callbacks for interactivity
@app.callback(
    [
        Output('fraud-trends', 'figure'),
        Output('fraud-device', 'figure'),
        Output('fraud-browser', 'figure'),
        Output('fraud-hour', 'figure'),
        Output('fraud-day', 'figure'),
        Output('alert-message', 'children'),
        Output('alert-message', 'style'),
    ],
    [
        Input('filter-button', 'n_clicks'),
        Input('reset-button', 'n_clicks'),
        Input('geo-distribution', 'clickData')
    ],
    [
        State('date-filter', 'start_date'),
        State('date-filter', 'end_date')
    ]
)

def update_charts(filter_clicks, reset_clicks, geo_click_data, start_date, end_date):
    ctx = callback_context
    filtered_data = fraud_data_processed
    alert_message = ""
    alert_style = {'display': 'none'}

    # Default behavior: no filters applied
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'reset-button.n_clicks':
        alert_message = "Displaying unfiltered data."
        alert_style = {'display': 'block', 'color': 'blue'}

    # Check which input triggered the callback
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'filter-button' and start_date and end_date:
            # Convert start_date and end_date to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_data = fraud_data_processed[
                (fraud_data_processed['purchase_time'] >= start_date) &
                (fraud_data_processed['purchase_time'] <= end_date)
            ]
            alert_message = f"Filters applied from {start_date.date()} to {end_date.date()}."
            alert_style = {'display': 'block', 'color': 'green'}

        elif button_id == 'geo-distribution' and geo_click_data:
            # Apply geographical filter if a country is clicked
            try:
                country = geo_click_data['points'][0]['location']
                filtered_data = filtered_data[filtered_data['country'] == country]
                alert_message = f"Filtered by country: {country}"
                alert_style = {'display': 'block', 'color': 'green'}
            except (KeyError, IndexError):
                alert_message = "Error retrieving country data."
                alert_style = {'display': 'block', 'color': 'red'}

    # Generate updated figures
    try:
        fraud_trends_fig = px.line(
            filtered_data.groupby(filtered_data['purchase_time'].dt.date)['class'].sum().reset_index(),
            x='purchase_time', y='class', template='plotly_white'
        ).update_traces(line_color='#e74c3c')

        fraud_device_fig = px.bar(
            filtered_data.groupby(['device_id', 'class']).size().unstack().fillna(0),
            template='plotly_white', color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        fraud_browser_fig = px.bar(
            filtered_data.groupby(['browser', 'class']).size().unstack().fillna(0),
            template='plotly_white', color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        fraud_hour_fig = px.bar(
            filtered_data[filtered_data['class'] == 1].groupby('purchase_hour').size(),
            template='plotly_white', color_discrete_sequence=['#e74c3c']
        ).update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Number of Fraud Cases'
        )

        fraud_day_fig = px.bar(
            filtered_data[filtered_data['class'] == 1].groupby('purchase_day').size(),
            template='plotly_white', color_discrete_sequence=['#e74c3c']
        ).update_layout(
            xaxis_title='Day of Week',
            yaxis_title='Number of Fraud Cases'
        )
    except Exception as e:
        print(f"Error generating figures: {e}")
        alert_message = "Error updating charts. Please check the data or filters."
        alert_style = {'display': 'block', 'color': 'red'}
        fraud_trends_fig = fraud_device_fig = fraud_browser_fig = fraud_hour_fig = fraud_day_fig = {}

    return fraud_trends_fig, fraud_device_fig, fraud_browser_fig, fraud_hour_fig, fraud_day_fig, alert_message, alert_style


# Initialize geographical distribution chart
@app.callback(
    Output('geo-distribution', 'figure'),
    Input('geo-distribution', 'id')
)
def initialize_geo_chart(_):
    geo_fig = px.choropleth(
        fraud_data_processed[fraud_data_processed['class'] == 1]
        .groupby('country')
        .size()
        .reset_index(name='count'),
        locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale='Reds',
        template='plotly_white',
        labels={'count': 'Fraud Cases'},
        hover_name='country'
    ).update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            landcolor='whitesmoke',
            lakecolor='white',
            showocean=True,
            oceancolor='aliceblue'
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(
            title="Fraud Cases",
            titlefont=dict(size=12),
            tickvals=[100, 1000, 3000, 4000, 5500],
            tickformat=',d',
            thickness=12,
            len=0.5
        )
    ).update_traces(
        hovertemplate="<b>Country:</b> %{location}<br><b>Fraud Cases:</b> %{z}<extra></extra>"
    )

    return geo_fig



app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>Fraud Detection Dashboard</title>
    {%favicon%}
    {%css%}
  <style>
    /* Navbar Styling */
    .navbar {
        background-color: #2c3e50;
        padding: 1rem;
        text-align: center;
        color: #ecf0f1;
    }
    .nav-title {
        font-family: 'Arial Black', sans-serif;
        font-size: 2rem;
        margin: 0;
    }
    .nav-subtitle {
        font-size: 1rem;
        margin-top: 5px;
        color: #bdc3c7;
        font-style: italic;
    }

    /* Main Content */
    .main-content {
        background-color: #f5f5f5;
        padding: 2rem;
    }

        /* Summary Statistics Cards */
        .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem; 
        margin-top: 2rem;
        }

        /* Individual stat card styling */
        /* Container for summary statistic cards */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem; /* Space between the cards */
        margin-top: 2rem; 
        padding: 0 1rem; 
    }

    /* Individual stat card styling */
    .stat-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        max-width: 400px; 
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    /* Hover effect for stat cards */
    .stat-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-8px); /* Lifts the card on hover */
    }

    /* Animation for the icon */
    @keyframes icon-bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-6px);
        }
    }

    .stat-icon {
        font-size: 2.5rem;
        color: #007bff;
        margin-bottom: 1rem;
        animation: icon-bounce 2s infinite; 
    }

    /* Enhanced styling for stat details */
    .stat-details {
        margin-top: 1rem;
    }

    .stat-label {
        font-weight: 600;
        color: #333;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        font-size: 1.4rem;
        color: #444;
    }

    .fraud-value {
        color: #d9534f;
        font-weight: bold;
    }

    /* Add a pulsing background on hover */
    .stat-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.5), transparent);
        transform: rotate(-45deg);
        transition: opacity 0.4s ease;
        opacity: 0;
        pointer-events: none;
    }

    .stat-card:hover::before {
        opacity: 0.5;
    }

    /* Fade-in animation for card load */
    @keyframes fadeIn {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stat-card {
        animation: fadeIn 0.8s ease forwards;
    }


    /* Charts Section */
    .charts-container {
        margin-top: 2rem;
    }
    .chart-title {
        font-size: 1.5rem;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    .chart-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        transition: box-shadow 0.3s;
        width: 100%; /* Default width */
    }
    .row stats-container {
        width: 100%;
        padding: 1.5rem;
        border-radius: 8px;
    }
    .chart-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .row {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 1rem;
    }
    .mb-4 {
        margin-bottom: 2rem;
    }

    /* Colors */
    .plotly_white .main-svg {
        background-color: #fbfbfb;
    }

    .col-md-6 {
        max-width: 45%; /* Wider width for specific charts */
    }

    .filters-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 1rem;
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    .date-picker {
        flex: 1;
    }

    .filter-button {
        padding: 0.5rem 1rem;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }

    .filter-button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    
</style>

</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)

