from dash import Dash, html, dcc
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
    fraud_data = pd.read_csv('dashboard/data/Fraud_Data.csv')
    credit_data = pd.read_csv('dashboard/data/render_creditcard.csv')
    ip_country = pd.read_csv('dashboard/data/IpAddress_to_Country.csv')
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

# Create the dashboard layout
app.layout = html.Div([
    # Navigation bar
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),
        html.P('fraud analytics and insights', className='nav-subtitle')
    ], className='navbar'),

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
                html.Div([
                    html.H3('Fraud Trends Over Time', className='chart-title'),
                    dcc.Graph(
                        figure=px.line(
                            fraud_data_processed.groupby(fraud_data_processed['purchase_time'].dt.date)['class'].sum().reset_index(),
                            x='purchase_time',
                            y='class',
                            template='plotly_white'
                        ).update_traces(line_color='#e74c3c')
                    )
                ], className='chart-card')
            ], className='mb-4'),

            html.Div([
                html.H3('Geographical Distribution of Fraud', className='chart-title'),
                dcc.Graph(
                    figure=px.choropleth(
                        fraud_data_processed[fraud_data_processed['class'] == 1].groupby('country').size().reset_index(name='count'),
                        locations='country',
                        locationmode='country names',
                        color='count',
                        color_continuous_scale='Reds',
                        template='plotly_white'
                    )
                )
            ], className='chart-card mb-4'),

            # Device and Browser Analysis
            html.Div([
                html.Div([
                    html.H3('Fraud by Device', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed.groupby(['device_id', 'class']).size().unstack().fillna(0),
                            template='plotly_white',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                    )
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Browser', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed.groupby(['browser', 'class']).size().unstack().fillna(0),
                            template='plotly_white',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                    )
                ], className='chart-card col-md-6')
            ], className='row mb-4'),

            # Time Patterns
            html.Div([
                html.Div([
                    html.H3('Fraud by Hour of Day', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_hour').size(),
                            template='plotly_white',
                            color_discrete_sequence=['#e74c3c']
                        ).update_layout(
                            xaxis_title='Hour of Day',
                            yaxis_title='Number of Fraud Cases'
                        )
                    )
                ], className='chart-card col-md-6'),

                html.Div([
                    html.H3('Fraud by Day of Week', className='chart-title'),
                    dcc.Graph(
                        figure=px.bar(
                            fraud_data_processed[fraud_data_processed['class'] == 1].groupby('purchase_day').size(),
                            template='plotly_white',
                            color_discrete_sequence=['#e74c3c']
                        ).update_layout(
                            xaxis_title='Day of Week',
                            yaxis_title='Number of Fraud Cases'
                        )
                    )
                ], className='chart-card col-md-6')
            ], className='row mb-4')
        ], className='charts-container')
    ], className='main-content')
])

# CSS and Enhanced Styling
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

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)
