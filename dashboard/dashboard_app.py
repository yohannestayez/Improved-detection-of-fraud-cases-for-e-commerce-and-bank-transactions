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
    """
    Converts an IPv4 address from its dotted-decimal string format to an integer.

    Parameters:
    ip (str): The IPv4 address in dotted-decimal string format (e.g., "192.168.1.1").

    Returns:
    int: The integer representation of the IPv4 address.
    None: If the input is not a valid IPv4 address.
    """
    try:
        # Convert the IP address string to a 32-bit packed binary format and unpack it as an integer.
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        # Return None if the IP address is invalid.
        return None


def load_data():
    """
    Loads datasets required for analysis or processing from CSV files.

    Returns:
    tuple: A tuple containing three pandas DataFrames:
        - fraud_data: DataFrame containing information about fraudulent activities.
        - credit_data: DataFrame containing credit card transaction data.
        - ip_country: DataFrame mapping IP addresses to country information.
    """
    # Load the fraud data dataset from the specified CSV file.
    fraud_data = pd.read_csv('data/Fraud_Data.csv')
    
    # Load the credit card transaction data from the specified CSV file.
    credit_data = pd.read_csv('data/render_creditcard.csv')
    
    # Load the IP-to-country mapping dataset from the specified CSV file.
    ip_country = pd.read_csv('data/IpAddress_to_Country.csv')
    
    # Return all three datasets as a tuple of DataFrames.
    return fraud_data, credit_data, ip_country


def process_ecommerce_data(fraud_data, ip_country):
    """
    Processes eCommerce data to clean, enrich, and integrate fraud data with IP-to-country mappings.

    Parameters:
    fraud_data (DataFrame): A pandas DataFrame containing fraud-related eCommerce data.
    ip_country (DataFrame): A pandas DataFrame mapping IP address ranges to countries.

    Returns:
    DataFrame: A cleaned and enriched DataFrame that includes fraud data with corresponding country information.
    """
    # Create a copy of the input DataFrames to avoid modifying the originals.
    fraud_data_cleaned = fraud_data.copy()
    ip_country_cleaned = ip_country.copy()

    # Convert signup and purchase time columns to datetime format.
    fraud_data_cleaned['signup_time'] = pd.to_datetime(fraud_data_cleaned['signup_time'])
    fraud_data_cleaned['purchase_time'] = pd.to_datetime(fraud_data_cleaned['purchase_time'])
    
    # Extract the day of the week and the hour of the purchase from the purchase time.
    fraud_data_cleaned['purchase_day'] = fraud_data_cleaned['purchase_time'].dt.day_name()
    fraud_data_cleaned['purchase_hour'] = fraud_data_cleaned['purchase_time'].dt.hour

    # Convert IP addresses to integer format using the ip_to_int function.
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_address'].apply(
        lambda x: ip_to_int(str(int(x))) if pd.notna(x) else None
    )

    # Ensure IP address range columns in the IP-country data are integers.
    ip_country_cleaned['lower_bound_ip_address'] = ip_country_cleaned['lower_bound_ip_address'].astype('int64')
    ip_country_cleaned['upper_bound_ip_address'] = ip_country_cleaned['upper_bound_ip_address'].astype('int64')
    fraud_data_cleaned['ip_int'] = fraud_data_cleaned['ip_int'].astype('int64')

    # Sort the IP-country data by the lower bound for efficient merging.
    ip_country_cleaned.sort_values('lower_bound_ip_address', inplace=True)

    # Use a merge_asof to map each fraud_data IP to the appropriate country based on IP ranges.
    fraud_data_with_country = pd.merge_asof(
        fraud_data_cleaned.sort_values('ip_int'),  # Sort fraud data by IP integer.
        ip_country_cleaned[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',  # Match based on the integer IP address.
        right_on='lower_bound_ip_address',  # Match against the lower bound of the IP range.
        direction='backward'  # Ensure backward merging to find the closest lower bound.
    )

    # Filter rows to ensure the IP integer falls within the valid IP range.
    fraud_data_with_country = fraud_data_with_country[
        (fraud_data_with_country['ip_int'] >= fraud_data_with_country['lower_bound_ip_address']) &
        (fraud_data_with_country['ip_int'] <= fraud_data_with_country['upper_bound_ip_address'])
    ]

    # Drop unnecessary columns related to IP address ranges after merging.
    fraud_data_with_country.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)

    # Return the processed DataFrame enriched with country information.
    return fraud_data_with_country

def create_summary_stats(fraud_data, credit_data):
    """
    Creates summary statistics for eCommerce fraud data and credit card transaction data.

    Parameters:
    fraud_data (DataFrame): A pandas DataFrame containing eCommerce fraud-related transactions.
                           Assumes the 'class' column indicates fraud (1 for fraud, 0 otherwise).
    credit_data (DataFrame): A pandas DataFrame containing credit card transaction data.
                             Assumes the 'Class' column indicates fraud (1 for fraud, 0 otherwise).

    Returns:
    tuple: A tuple containing two dictionaries:
        - ecom_stats: Summary statistics for the eCommerce fraud data.
        - credit_stats: Summary statistics for the credit card transaction data.
    """
    # Calculate summary statistics for the eCommerce fraud data.
    ecom_stats = {
        'total_transactions': len(fraud_data),  # Total number of transactions in the dataset.
        'fraud_cases': fraud_data['class'].sum(),  # Total number of fraudulent transactions.
        'fraud_percentage': (fraud_data['class'].sum() / len(fraud_data) * 100).round(2)  # Percentage of fraud cases.
    }

    # Calculate summary statistics for the credit card transaction data.
    credit_stats = {
        'total_transactions': len(credit_data),  # Total number of transactions in the dataset.
        'fraud_cases': credit_data['Class'].sum(),  # Total number of fraudulent transactions.
        'fraud_percentage': (credit_data['Class'].sum() / len(credit_data) * 100).round(2)  # Percentage of fraud cases.
    }

    # Return the summary statistics for both datasets.
    return ecom_stats, credit_stats

# Load and process data
fraud_data, credit_data, ip_country = load_data()
fraud_data_processed = process_ecommerce_data(fraud_data, ip_country)
ecom_stats, credit_stats = create_summary_stats(fraud_data_processed, credit_data)

# App layout definition
app.layout = html.Div([
    # Navigation bar at the top
    html.Div([
        html.H1('Fraud Detection Dashboard', className='nav-title'),  # Dashboard title
        html.P('fraud analytics and insights', className='nav-subtitle')  # Subtitle
    ], className='navbar'),  # Apply navbar styling

    # Filters section for selecting date ranges and resetting filters
    html.Div([
        # Date picker to filter transactions by date range
        dcc.DatePickerRange(
            id='date-filter',  # Unique ID for the date picker
            start_date=fraud_data_processed['purchase_time'].min(),  # Set start date to earliest purchase time
            end_date=fraud_data_processed['purchase_time'].max(),  # Set end date to latest purchase time
            display_format='YYYY-MM-DD',  # Format for displayed dates
            className='date-picker'  # Apply date picker styling
        ),
        # Button to apply filters
        html.Button('Filter', id='filter-button', className='filter-button'),
        # Button to reset filters
        html.Button('Reset Filters', id='reset-button', className='filter-button')
    ], className='filters-container'),  # Container for filters

    # Alert message container, hidden by default
    html.Div(id='alert-message', className='alert-message', style={'display': 'none'}),

    # Main content container
    html.Div([
        # Summary statistics cards for displaying transaction summaries
        html.Div([
            # eCommerce transactions summary card
            html.Div([
                html.Div([
                    html.I(className='fas fa-shopping-cart stat-icon'),  # Icon for eCommerce
                    html.Div([
                        html.H3('E-commerce Transactions'),  # Title for the section
                        html.Div([
                            # Total transactions
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{ecom_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            # Fraud cases
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            # Fraud percentage
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{ecom_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')  # Styling for the card
            ], className='col-md-6'),  # Layout for eCommerce section

            # Credit card transactions summary card
            html.Div([
                html.Div([
                    html.I(className='fas fa-credit-card stat-icon'),  # Icon for credit cards
                    html.Div([
                        html.H3('Credit Card Transactions'),  # Title for the section
                        html.Div([
                            # Total transactions
                            html.P([
                                html.Span('Total Transactions: ', className='stat-label'),
                                html.Span(f"{credit_stats['total_transactions']:,}", className='stat-value')
                            ]),
                            # Fraud cases
                            html.P([
                                html.Span('Fraud Cases: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_cases']:,}", className='stat-value fraud-value')
                            ]),
                            # Fraud percentage
                            html.P([
                                html.Span('Fraud Percentage: ', className='stat-label'),
                                html.Span(f"{credit_stats['fraud_percentage']}%", className='stat-value fraud-value')
                            ])
                        ], className='stat-details')
                    ])
                ], className='stat-card')  # Styling for the card
            ], className='col-md-6')  # Layout for credit card section
        ], className='row stats-container'),  # Row container for statistics

        # Charts section for visualizations
        html.Div([
            # Fraud trends over time
            html.Div([
                html.H3('Fraud Trends Over Time', className='chart-title'),  # Title
                dcc.Graph(id='fraud-trends')  # Placeholder for the fraud trends graph
            ], className='chart-card mb-4'),

            # Geographical distribution of fraud
            html.Div([
                html.H3(
                    'Geographical Distribution of Fraud', 
                    className='chart-title', 
                    style={'textAlign': 'center', 'marginBottom': '15px'}
                ),
                dcc.Graph(id='geo-distribution', className='chart-card mb-4')  # Placeholder for the geographical distribution graph
            ], style={
                'padding': '20px', 
                'backgroundColor': 'white', 
                'borderRadius': '8px', 
                'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
            }),

            # Fraud analysis by device and browser
            html.Div([
                # Fraud by device
                html.Div([
                    html.H3('Fraud by Device', className='chart-title'),  # Title
                    dcc.Graph(id='fraud-device')  # Placeholder for the device graph
                ], className='chart-card col-md-6'),

                # Fraud by browser
                html.Div([
                    html.H3('Fraud by Browser', className='chart-title'),  # Title
                    dcc.Graph(id='fraud-browser')  # Placeholder for the browser graph
                ], className='chart-card col-md-6')
            ], className='row mb-4'),

            # Time patterns in fraud
            html.Div([
                # Fraud by hour of day
                html.Div([
                    html.H3('Fraud by Hour of Day', className='chart-title'),  # Title
                    dcc.Graph(id='fraud-hour')  # Placeholder for the hour graph
                ], className='chart-card col-md-6'),

                # Fraud by day of week
                html.Div([
                    html.H3('Fraud by Day of Week', className='chart-title'),  # Title
                    dcc.Graph(id='fraud-day')  # Placeholder for the day graph
                ], className='chart-card col-md-6')
            ], className='row mb-4')
        ], className='charts-container')  # Container for charts
    ], className='main-content')  # Main content styling
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
    """
    Updates the charts displayed in the dashboard based on user interactions such as applying filters,
    resetting filters, or selecting geographical data. Returns updated charts and an alert message.

    Args:
        filter_clicks (int): Number of clicks on the filter button.
        reset_clicks (int): Number of clicks on the reset button.
        geo_click_data (dict): Data from a geographical chart click event.
        start_date (str): Start date for filtering in 'YYYY-MM-DD' format.
        end_date (str): End date for filtering in 'YYYY-MM-DD' format.

    Returns:
        tuple: Contains updated figures for charts, alert message, and alert style.
    """

    # Initialize context and defaults
    ctx = callback_context
    filtered_data = fraud_data_processed  # Default to the entire dataset
    alert_message = ""  # Message to be displayed as a user alert
    alert_style = {'display': 'none'}  # Default style for alert (hidden)

    # Default behavior: No filters applied, show unfiltered data
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'reset-button.n_clicks':
        alert_message = "Displaying unfiltered data."
        alert_style = {'display': 'block', 'color': 'blue'}

    # Check which input triggered the callback
    if ctx.triggered:
        # Extract the triggering component's ID
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Apply date range filter if the filter button is clicked and dates are provided
        if button_id == 'filter-button' and start_date and end_date:
            # Convert start_date and end_date to datetime objects
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # Filter data based on the date range
            filtered_data = fraud_data_processed[
                (fraud_data_processed['purchase_time'] >= start_date) &
                (fraud_data_processed['purchase_time'] <= end_date)
            ]

            # Update alert message to indicate successful filtering
            alert_message = f"Filters applied from {start_date.date()} to {end_date.date()}."
            alert_style = {'display': 'block', 'color': 'green'}

        # Apply geographical filter if a country is clicked on the map
        elif button_id == 'geo-distribution' and geo_click_data:
            try:
                # Extract the clicked country's location
                country = geo_click_data['points'][0]['location']

                # Filter data based on the selected country
                filtered_data = filtered_data[filtered_data['country'] == country]
                alert_message = f"Filtered by country: {country}"
                alert_style = {'display': 'block', 'color': 'green'}
            except (KeyError, IndexError):
                # Handle errors if no valid country data is found in the click event
                alert_message = "Error retrieving country data."
                alert_style = {'display': 'block', 'color': 'red'}

    # Generate updated figures based on the filtered data
    try:
        # Line chart for fraud trends over time
        fraud_trends_fig = px.line(
            filtered_data.groupby(filtered_data['purchase_time'].dt.date)['class'].sum().reset_index(),
            x='purchase_time', y='class', template='plotly_white'
        ).update_traces(line_color='#e74c3c')  # Highlight fraud cases in red

        # Bar chart for fraud cases by device ID
        fraud_device_fig = px.bar(
            filtered_data.groupby(['device_id', 'class']).size().unstack().fillna(0),
            template='plotly_white', color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        # Bar chart for fraud cases by browser
        fraud_browser_fig = px.bar(
            filtered_data.groupby(['browser', 'class']).size().unstack().fillna(0),
            template='plotly_white', color_discrete_sequence=['#2ecc71', '#e74c3c']
        )

        # Bar chart for fraud cases by hour of the day
        fraud_hour_fig = px.bar(
            filtered_data[filtered_data['class'] == 1].groupby('purchase_hour').size(),
            template='plotly_white', color_discrete_sequence=['#e74c3c']
        ).update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Number of Fraud Cases'
        )

        # Bar chart for fraud cases by day of the week
        fraud_day_fig = px.bar(
            filtered_data[filtered_data['class'] == 1].groupby('purchase_day').size(),
            template='plotly_white', color_discrete_sequence=['#e74c3c']
        ).update_layout(
            xaxis_title='Day of Week',
            yaxis_title='Number of Fraud Cases'
        )
    except Exception as e:
        # Handle errors during figure generation
        print(f"Error generating figures: {e}")
        alert_message = "Error updating charts. Please check the data or filters."
        alert_style = {'display': 'block', 'color': 'red'}

        # Return empty figures in case of error
        fraud_trends_fig = fraud_device_fig = fraud_browser_fig = fraud_hour_fig = fraud_day_fig = {}

    # Return all updated charts, alert message, and alert style
    return fraud_trends_fig, fraud_device_fig, fraud_browser_fig, fraud_hour_fig, fraud_day_fig, alert_message, alert_style


# Initialize geographical distribution chart
@app.callback(
    Output('geo-distribution', 'figure'),
    Input('geo-distribution', 'id')
)
def initialize_geo_chart(_):
    """
    Initializes a geographical chart (choropleth map) that visualizes fraud cases by country.
    
    Args:
        _ (any): Placeholder for callback input, not used in this function.
        
    Returns:
        geo_fig (plotly.graph_objs._figure.Figure): Choropleth map showing fraud cases by country.
    """

    # Create a choropleth map using fraud data filtered for fraudulent transactions (class == 1)
    geo_fig = px.choropleth(
        # Group data by country and count the number of fraud cases
        fraud_data_processed[fraud_data_processed['class'] == 1]
        .groupby('country')
        .size()
        .reset_index(name='count'),  # Rename the grouped column to 'count'

        # Specify the column representing countries
        locations='country',
        locationmode='country names',  # Use country names as identifiers for locations

        # Specify the column representing the fraud case count
        color='count',

        # Define the color scale for the choropleth
        color_continuous_scale='Reds',  # Red tones to highlight fraud intensity

        # Set a default template for the plot
        template='plotly_white',

        # Label the color legend
        labels={'count': 'Fraud Cases'},

        # Use the country name for hover information
        hover_name='country'
    ).update_layout(
        # Configure map appearance
        geo=dict(
            showframe=False,             # Remove the map frame
            showcoastlines=True,         # Display coastlines
            projection_type='natural earth',  # Use natural earth projection
            landcolor='whitesmoke',      # Set land color
            lakecolor='white',           # Set lake color
            showocean=True,              # Display ocean
            oceancolor='aliceblue'       # Set ocean color
        ),

        # Adjust the chart margins
        margin=dict(l=10, r=10, t=40, b=10),

        # Customize the color axis colorbar
        coloraxis_colorbar=dict(
            title="Fraud Cases",         # Title for the colorbar
            titlefont=dict(size=12),     # Font size for the title
            tickvals=[100, 1000, 3000, 4000, 5500],  # Custom tick values
            tickformat=',d',             # Format tick values with commas for readability
            thickness=12,                # Thickness of the colorbar
            len=0.5                      # Length of the colorbar as a fraction of the plot height
        )
    ).update_traces(
        # Configure hover information template for each country
        hovertemplate="<b>Country:</b> %{location}<br><b>Fraud Cases:</b> %{z}<extra></extra>"
    )

    # Return the generated geographical chart
    return geo_fig



app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}  <!-- Inserts meta tags for the app, e.g., viewport settings for mobile responsiveness -->
    <title>Fraud Detection Dashboard</title>  <!-- Sets the title of the app that appears in the browser tab -->
    {%favicon%}  <!-- Adds the favicon for the app -->
    {%css%}  <!-- Links to the app's CSS stylesheets -->
</head>
<body>
    {%app_entry%}  <!-- The main content of the Dash app is rendered here -->
    <footer>
        {%config%}  <!-- Inserts Dash-specific configuration settings -->
        {%scripts%}  <!-- Adds the JavaScript scripts required for the app to function -->
        {%renderer%}  <!-- Configures the renderer for the app (e.g., React-based rendering) -->
    </footer>
</body>
</html>
'''


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)

