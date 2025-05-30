import openai
import json
import pandas as pd
import streamlit as st
from secret_key import openai_key  # Make sure this file has: openai_key = "sk-..."

# Set API key
openai.api_key = openai_key

# Prompt for financial data extraction
def get_prompt_financial():
    return '''Please retrieve company name, revenue, net income and earnings per share (a.k.a. EPS)
from the following news article. If you can't find the information from this article 
then return "". Do not make things up.    
Then retrieve a stock symbol corresponding to that company. For this you can use
your general knowledge (it doesn't have to be from this article). Always return your
response as a valid JSON string. The format of that string should be:
{
    "Company Name": "Walmart",
    "Stock Symbol": "WMT",
    "Revenue": "12.34 million",
    "Net Income": "34.78 million",
    "EPS": "2.1 $"
}
News Article:
=============
'''

# Function to extract and parse data
def extract_financial_data(text):
    prompt = get_prompt_financial() + text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['choices'][0]['message']['content']

        # Show raw GPT response for debugging
        st.markdown("### 🧪 Raw GPT Response:")
        st.code(content, language="json")

        # Extract JSON from response safely
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]

        data = json.loads(json_str)
        return pd.DataFrame(data.items(), columns=["Measure", "Value"])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return pd.DataFrame({
            "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
            "Value": ["", "", "", "", ""]
        })

# --- Streamlit UI ---
st.set_page_config(page_title="Financial Extractor", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .stTextArea textarea {
            font-size: 16px;
            line-height: 1.6;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Financial News Data Extractor")
st.markdown("##### 💬 Paste a financial news article below to extract structured data.")

# Input column
col1, col2 = st.columns([3, 2])

with col1:
    news_article = st.text_area(
        label="📰 News Article",
        height=250,
        placeholder="Example: Apple reported $94.8B in revenue and $24.1B in net income this quarter..."
    )

    if st.button("🚀 Extract Financial Data"):
        if news_article.strip() == "":
            st.warning("Please paste a news article first.")
        else:
            financial_data_df = extract_financial_data(news_article)
    else:
        financial_data_df = pd.DataFrame({
            "Measure": ["Company Name", "Stock Symbol", "Revenue", "Net Income", "EPS"],
            "Value": ["", "", "", "", ""]
        })

# Output column
with col2:
    st.markdown("<br/><br/><br/><br/><br/>", unsafe_allow_html=True)
    st.subheader("📈 Extracted Financial Data")
    st.dataframe(
        financial_data_df,
        column_config={
            "Measure": st.column_config.Column(width=150),
            "Value": st.column_config.Column(width=150)
        },
        hide_index=True,
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("🔍 Powered by OpenAI GPT-3.5 Turbo")
