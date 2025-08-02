import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(
    page_title="RentSage - All-in-One Platform for Rental Insights Across Canada",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stApp {
    background: 
        linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
        url('https://i.imgur.com/nTiJ4HR.jpeg');
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    }
        div[data-testid="stTitle"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    h1 {
    color: white;
    padding: 15px 25px;  
    border-radius: 10px;
    display: inline-block;
    text-align: center;
    letter-spacing: 2px;
    font-family: 'Orbitron', 'Courier New', monospace;
    font-weight: 700;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
* {
    font-family: 'Montserrat', 'Helvetica', sans-serif;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def scrape_zumper_data():
    """
    Scrape rental data from Zumper Canada report
    Returns: dict with image_url and dataframe
    """
    url = "https://zumper.com/blog/rental-price-data-canada"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    
    chart_img = soup.find('img', alt="Year-over-year price changes to the Canadian national one and two-bedroom rents\n")
    if not chart_img: 
        chart_img = soup.find('img', alt="Year-over-year price changes to the Canadian national one and two-bedroom rents")

    if chart_img:
      image_url = chart_img['src'] 
    else: 
        image_url = None
    
    # Extract table data
    table = soup.find('table', id='tablepress-786')
    
    if not table:
        print("Table not found")
        return None
    
    # Extract table rows
    rows = []
    for tr in table.find('tbody').find_all('tr')[1:]:  # Skip header row
        row_data = []
        for td in tr.find_all('td'):
            # Clean up the text and extract city name from links
            if td.find('a'):
                city_text = td.find('a').get_text(strip=True)
                row_data.append(city_text)
            else:
                text = td.get_text(strip=True)
                row_data.append(text)
        rows.append(row_data)
    
    # Create DataFrame
    columns = [
        'Ranking', 'Ranking_Change', 'City', 
        '1BR_Price', '1BR_Monthly_Change', '1BR_Yearly_Change',
        '2BR_Price', '2BR_Monthly_Change', '2BR_Yearly_Change'
    ]
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Clean up the data
    df['1BR_Price'] = df['1BR_Price'].str.replace('$', '').str.replace(',', '').astype(int)
    df['2BR_Price'] = df['2BR_Price'].str.replace('$', '').str.replace(',', '').astype(int)
    df['1BR_Monthly_Change'] = df['1BR_Monthly_Change'].str.replace('%', '').astype(float)
    df['1BR_Yearly_Change'] = df['1BR_Yearly_Change'].str.replace('%', '').astype(float)
    df['2BR_Monthly_Change'] = df['2BR_Monthly_Change'].str.replace('%', '').astype(float)
    df['2BR_Yearly_Change'] = df['2BR_Yearly_Change'].str.replace('%', '').astype(float)
    df['Ranking'] = df['Ranking'].astype(int)
    
    return {
        'image_url': image_url,
        'rental_data': df
    }

pred_model = joblib.load('rentalpricemodel.joblib')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)
with open('categorical_options.json', 'r') as f:
    categorical_options = json.load(f)
    province_cities = categorical_options['province_cities']


st.title("RentSage")

tab1, tab2 = st.tabs(['Rent Prediction', 'Market Analysis'])

with tab1: 
        
    st.sidebar.header("Prediction Tool")

    beds = st.sidebar.selectbox("Bedrooms", options=[0, 1, 2, 3, 4, 5])
    baths = st.sidebar.selectbox("Bathrooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    sq_feet = st.sidebar.number_input("Square Feet", min_value=300, value=800)
    cats = st.sidebar.checkbox("Cats Allowed")
    dogs = st.sidebar.checkbox("Dogs Allowed")
    province = st.sidebar.selectbox("Province", options=list(province_cities.keys()),index=0)
    city = st.sidebar.selectbox("City", options=province_cities[province])
    lease_term = st.sidebar.selectbox("Lease Term", options=categorical_options['lease_term'])
    prop_type = st.sidebar.selectbox("Property Type", options=categorical_options['type'])
    furnishing = st.sidebar.selectbox("Furnishing", options=categorical_options['furnishing'])
    availability = st.sidebar.selectbox("Availability", options=categorical_options['availability_date'])
    smoking = st.sidebar.selectbox("Smoking", options=categorical_options['smoking'])

    def preprocessing():

        pet_eligibility = int(cats and dogs)
        total_rooms = beds + baths

        log_beds = np.log1p(beds)
        log_baths = np.log1p(baths)
        log_sqfeet = np.log1p(sq_feet)
        log_totalrooms = np.log1p(total_rooms)

        input_dict = {
            'beds': log_beds,
            'baths': log_baths,
            'sq_feet': log_sqfeet,
            'total_rooms': log_totalrooms,
            'pet_eligibility': pet_eligibility,
            'city': city,
            'province': province,
            'lease_term': lease_term,
            'type': prop_type,
            'furnishing': furnishing,
            'availability_date': availability,
            'smoking': smoking
        }

        input_df = pd.DataFrame([input_dict])
        
        
        categorical_features = ['city', 'province', 'lease_term', 'type', 'furnishing', 'availability_date', 'smoking']
        
        for category in categorical_features:
            if category in input_df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(input_df[category], prefix=category, drop_first=True)
                
                for option in categorical_options[category]:
                    col_name = f"{category}_{option}"
                    if col_name not in dummies.columns and col_name in feature_columns:
                        dummies[col_name] = 0
                
                input_df = pd.concat([input_df, dummies], axis=1)
                input_df.drop(category, axis=1, inplace=True)
        
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        
        return input_df[feature_columns]

    if st.sidebar.button("Predict Rent"):
        try:
            input_processed = preprocessing()
            prediction = pred_model.predict(input_processed)[0]
            st.success(f"## Predicted Monthly Rent: ${prediction:,.2f}")
            st.balloons()
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

with tab2: 
    st.header("📊 Canadian Rental Market Analysis")
    st.markdown("*Data courtesy of [Zumper](https://zumper.com/blog/rental-price-data-canada) - Canadian Rent Report*")
    
    zumper_data = scrape_zumper_data()
    
    if zumper_data:
        # Display the chart
        if zumper_data['image_url']:
            st.subheader("📈 Year-over-Year Price Changes")
            st.image(zumper_data['image_url'], caption="Canadian National Rent Index", use_container_width=True)
        
        # Display the data table
        st.subheader("🏙️ Top Canadian Cities - Rental Prices")
        
        df = zumper_data['rental_data']
        
        # Add some metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_1br = df['1BR_Price'].mean()
            st.metric("Average 1BR Rent", f"${avg_1br:,.0f}")
        
        with col2:
            avg_2br = df['2BR_Price'].mean()
            st.metric("Average 2BR Rent", f"${avg_2br:,.0f}")
        
        with col3:
            highest_city = df.iloc[0]['City']
            highest_price = df.iloc[0]['1BR_Price']
            st.metric("Most Expensive (1BR)", f"{highest_city}", f"${highest_price:,}")
        
        with col4:
            lowest_city = df.iloc[-1]['City']
            lowest_price = df.iloc[-1]['1BR_Price']
            st.metric("Most Affordable (1BR)", f"{lowest_city}", f"${lowest_price:,}")
        
        # Display searchable/sortable table
        st.dataframe(
            df.style.format({
                '1BR_Price': '${:,}',
                '2BR_Price': '${:,}',
                '1BR_Monthly_Change': '{:.1f}%',
                '1BR_Yearly_Change': '{:.1f}%',
                '2BR_Monthly_Change': '{:.1f}%',
                '2BR_Yearly_Change': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        st.subheader("🔍 Market Insights")
        
        # Cities with highest growth
        top_growth = df.nlargest(3, '1BR_Yearly_Change')
        st.write("**Cities with Highest 1BR Rent Growth (Year-over-Year):**")
        for idx, row in top_growth.iterrows():
            st.write(f"• {row['City']}: +{row['1BR_Yearly_Change']:.1f}%")
        
        # Cities with biggest declines
        top_decline = df.nsmallest(3, '1BR_Yearly_Change')
        st.write("**Cities with Biggest 1BR Rent Declines (Year-over-Year):**")
        for idx, row in top_decline.iterrows():
            st.write(f"• {row['City']}: {row['1BR_Yearly_Change']:.1f}%")
            
    else:
        st.error("Unable to load market data at this time.")