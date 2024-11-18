from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from flask import Flask
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

# Create the dashboard layout
app.layout = html.Div([
    # JavaScript for alerts
    html.Script("""
        function showAlert(message) {
            alert(message);
        }
    """),

    # Navigation bar
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),
        html.P('Fraud analytics and insights', className='nav-subtitle')
    ], className='navbar'),

    # Filters
    html.Div([
        html.Label("Filter by Country:"),
        dcc.Dropdown(
            id='country-filter',
            options=[{'label': country, 'value': country} for country in fraud_data_processed['country'].dropna().unique()],
            placeholder="Select a country"
        )
    ], className='filter-container'),

    # Summary Statistics Cards
    html.Div([
        html.Div([
            html.H3('E-commerce Transactions'),
            html.P(f"Total Transactions: {ecom_stats['total_transactions']:,}"),
            html.P(f"Fraud Cases: {ecom_stats['fraud_cases']:,}"),
            html.P(f"Fraud Percentage: {ecom_stats['fraud_percentage']}%")
        ], className='stat-card'),

        html.Div([
            html.H3('Credit Card Transactions'),
            html.P(f"Total Transactions: {credit_stats['total_transactions']:,}"),
            html.P(f"Fraud Cases: {credit_stats['fraud_cases']:,}"),
            html.P(f"Fraud Percentage: {credit_stats['fraud_percentage']}%")
        ], className='stat-card')
    ], className='stats-container'),

    # Charts Section
    html.Div([
        dcc.Graph(id='fraud-trends'),
        dcc.Graph(id='fraud-map')
    ], className='charts-container')
])

# Callbacks for interactivity
@app.callback(
    [Output('fraud-trends', 'figure'),
     Output('fraud-map', 'figure')],
    [Input('country-filter', 'value')]
)
def update_charts(selected_country):
    if selected_country:
        filtered_data = fraud_data_processed[fraud_data_processed['country'] == selected_country]
        js_alert = f"showAlert('Filtered by country: {selected_country} with {len(filtered_data)} records.');"
    else:
        filtered_data = fraud_data_processed
        js_alert = f"showAlert('Showing data for all countries.');"

    # Execute the JavaScript alert
    app.layout.children.append(html.Script(js_alert))

    # Update the fraud trends chart
    fraud_trends = px.line(
        filtered_data.groupby(filtered_data['purchase_time'].dt.date)['class'].sum().reset_index(),
        x='purchase_time',
        y='class',
        title='Fraud Trends Over Time'
    )

    # Update the fraud map
    fraud_map = px.choropleth(
        filtered_data[filtered_data['class'] == 1].groupby('country').size().reset_index(name='count'),
        locations='country',
        locationmode='country names',
        color='count',
        title='Geographical Distribution of Fraud',
        color_continuous_scale='Reds'
    )

    return fraud_trends, fraud_map

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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
