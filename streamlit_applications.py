import streamlit as st
import pandas as pd
from langchain_community.llms import Cohere  # Corrected import
import cohere
import re

# Load sentiment data (your dataset)
sentiment_data = pd.read_csv('C:/Users/Admin/Downloads/sentimental score dataset.csv')

# Initialize Cohere (Replace with your Cohere API Key)
cohere_api_key = 'your API key'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

# Add custom background and styling
st.markdown(
    """
    <style>
    body {
        background-color: black;
    }
    .main-title {
        color: blue;
        text-align: center;
        font-size: 36px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .recommendation {
        color: blue;
        font-size: 20px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid blue;
        border-radius: 5px;
        background-color: #1c1c1c;
    }
    a {
        color: cyan;
        text-decoration: none;
    }
    a:hover {
        color: lightblue;
        text-decoration: underline;
    }
    .input-box {
        background-color: #333333;
        color: white;
        border: 1px solid blue;
        padding: 5px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to recommend products based on user query
def recommend_products(query, sentiment_data):
    # Analyze the user's input using Cohere
    response = co.generate(prompt=query, model='command-xlarge-nightly')

    # Process response (you can also use a custom NLP method if needed)
    search_keywords = response.generations[0].text.strip().split()

    # Filter out invalid or empty search keywords
    search_keywords = [keyword for keyword in search_keywords if keyword and re.match(r'^\w+$', keyword)]

    if not search_keywords:
        return pd.DataFrame()  # Return empty DataFrame if no valid keywords

    # Create regex pattern from search keywords
    regex_pattern = '|'.join(search_keywords)

    # Filter sentiment data based on query analysis (e.g., find related product names or positive reviews)
    recommendations = sentiment_data[sentiment_data['Cleaned_Review'].str.contains(regex_pattern, case=False, na=False)]

    # Remove duplicate product_id rows
    recommendations = recommendations.drop_duplicates(subset='product_id')

    # Sort products by price (now using Price column instead of Sentiment_Score)
    recommendations = recommendations.sort_values(by='Price', ascending=False)

    # Return top 5 products
    return recommendations[['product_id', 'Price']].head(5)

# Streamlit App Interface
st.markdown("<h1 class='main-title'>Product Recommendation Engine</h1>", unsafe_allow_html=True)

user_query = st.text_input("Describe the type of product you're looking for (e.g., best camera phone, lightweight mobiles, budget-friendly options):", key="query_input")

if user_query:
    # Get top 5 product recommendations
    recommended_products = recommend_products(user_query, sentiment_data)

    if not recommended_products.empty:
        st.markdown("<h3 class='main-title'>Top 5 Recommended Products:</h3>", unsafe_allow_html=True)

        # Display each recommended product with rank
        for idx, row in enumerate(recommended_products.iterrows(), 1):
            product_id = row[1]['product_id']
            price = row[1]['Price']
            
            st.markdown(
                f"""
                <div class='recommendation'>
                    <strong>Rank {idx}:</strong> <br>
                    <strong>Product ID:</strong> {product_id} <br>
                    <strong>Price:</strong> ₹{price}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Option to open recommendations in a new tab
        st.markdown(
            f'<a href="https://www.flipkart.com/search?q={user_query}" target="_blank">Check products on Flipkart</a>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='recommendation'>No products found based on your query.</div>", unsafe_allow_html=True)
