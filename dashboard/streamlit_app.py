import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.database import DatabaseManager
from app.services.ml_service import MLService
from app.services.rag_service import RAGService
from app.services.agent_assist_service import AgentAssistService

# Page config
st.set_page_config(
    page_title="Contact Center AI Dashboard",
    page_icon="📞",
    layout="wide"
)

# Initialize services
@st.cache_resource
def init_services():
    return {
        'db': DatabaseManager(),
        'ml': MLService(),
        'rag': RAGService(),
        'agent_assist': AgentAssistService()
    }

services = init_services()

# Sidebar
st.sidebar.title("📞 Contact Center AI")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Analytics Dashboard", "Live Demo", "Sample Conversations"]
)

if page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    
    # Get analytics data
    try:
        analytics_data = services['db'].get_analytics_data()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", analytics_data['total_conversations'])
        
        with col2:
            if analytics_data['intent_distribution']:
                top_intent = max(analytics_data['intent_distribution'], 
                               key=analytics_data['intent_distribution'].get)
                st.metric("Top Intent", top_intent)
            else:
                st.metric("Top Intent", "N/A")
        
        with col3:
            if analytics_data['sentiment_distribution']:
                positive_pct = analytics_data['sentiment_distribution'].get('positive', 0)
                total = sum(analytics_data['sentiment_distribution'].values())
                positive_rate = (positive_pct / total * 100) if total > 0 else 0
                st.metric("Positive Sentiment %", f"{positive_rate:.1f}%")
            else:
                st.metric("Positive Sentiment %", "0%")
        
        with col4:
            if analytics_data['daily_volume']:
                avg_daily = sum(analytics_data['daily_volume'].values()) / len(analytics_data['daily_volume'])
                st.metric("Avg Daily Volume", f"{avg_daily:.0f}")
            else:
                st.metric("Avg Daily Volume", "0")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Intent Distribution")
            if analytics_data['intent_distribution']:
                intent_df = pd.DataFrame(
                    list(analytics_data['intent_distribution'].items()),
                    columns=['Intent', 'Count']
                )
                fig = px.pie(intent_df, values='Count', names='Intent', 
                           title="Customer Intent Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No intent data available")
        
        with col2:
            st.subheader("Sentiment Distribution")
            if analytics_data['sentiment_distribution']:
                sentiment_df = pd.DataFrame(
                    list(analytics_data['sentiment_distribution'].items()),
                    columns=['Sentiment', 'Count']
                )
                colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
                fig = px.bar(sentiment_df, x='Sentiment', y='Count',
                           color='Sentiment', color_discrete_map=colors,
                           title="Sentiment Analysis Results")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        # Daily volume chart
        st.subheader("Daily Conversation Volume")
        if analytics_data['daily_volume']:
            daily_df = pd.DataFrame(
                list(analytics_data['daily_volume'].items()),
                columns=['Date', 'Count']
            )
            daily_df['Date'] = pd.to_datetime(daily_df['Date'])
            fig = px.line(daily_df, x='Date', y='Count',
                         title="Daily Conversation Volume Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily volume data available")
            
    except Exception as e:
        st.error(f"Error loading analytics data: {e}")

elif page == "Live Demo":
    st.title("🤖 Live AI Demo")
    
    tab1, tab2, tab3 = st.tabs(["Intent & Sentiment", "RAG Query", "Agent Assist"])
    
    with tab1:
        st.subheader("Intent Classification & Sentiment Analysis")
        
        message = st.text_area("Enter a customer message:", 
                              placeholder="e.g., I want to return this product and get my money back")
        
        if st.button("Analyze Message"):
            if message:
                try:
                    # Predict intent and sentiment
                    intent, intent_conf = services['ml'].predict_intent(message)
                    sentiment, sentiment_conf = services['ml'].predict_sentiment(message)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**Intent:** {intent}")
                        st.info(f"Confidence: {intent_conf:.2f}")
                    
                    with col2:
                        color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                        st.success(f"**Sentiment:** {color.get(sentiment, '')} {sentiment}")
                        st.info(f"Confidence: {sentiment_conf:.2f}")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
            else:
                st.warning("Please enter a message to analyze")
    
    with tab2:
        st.subheader("Knowledge Base Query (RAG)")
        
        question = st.text_input("Ask a question about our policies:",
                                placeholder="e.g., What is your refund policy?")
        
        if st.button("Search Knowledge Base"):
            if question:
                try:
                    result = services['rag'].query(question)
                    
                    st.success("**Answer:**")
                    st.write(result['answer'])
                    
                    if result['sources']:
                        st.info("**Sources:** " + ", ".join(result['sources']))
                        
                except Exception as e:
                    st.error(f"Query failed: {e}")
            else:
                st.warning("Please enter a question")
    
    with tab3:
        st.subheader("Agent Assist")
        
        customer_msg = st.text_area("Customer message:",
                                   placeholder="Enter the customer's message to get AI assistance")
        
        if st.button("Get Agent Assistance"):
            if customer_msg:
                try:
                    result = services['agent_assist'].assist_agent(customer_msg)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Detected Intent:** {result['detected_intent']}")
                        st.info(f"**Sentiment:** {result['sentiment']}")
                    
                    with col2:
                        st.success("**Suggested Reply:**")
                        st.write(result['suggested_reply'])
                        
                except Exception as e:
                    st.error(f"Agent assist failed: {e}")
            else:
                st.warning("Please enter a customer message")

elif page == "Sample Conversations":
    st.title("💬 Sample Conversations")
    
    try:
        df = services['db'].get_all_conversations()
        
        if len(df) > 0:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                intent_filter = st.selectbox("Filter by Intent", 
                                           ["All"] + list(df['intent'].unique()))
            
            with col2:
                sentiment_filter = st.selectbox("Filter by Sentiment",
                                              ["All"] + list(df['sentiment'].unique()))
            
            with col3:
                num_records = st.slider("Number of records", 5, 50, 10)
            
            # Apply filters
            filtered_df = df.copy()
            if intent_filter != "All":
                filtered_df = filtered_df[filtered_df['intent'] == intent_filter]
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
            
            # Display conversations
            st.subheader(f"Showing {min(num_records, len(filtered_df))} conversations")
            
            for idx, row in filtered_df.head(num_records).iterrows():
                with st.expander(f"Conversation {row['id'][:8]} - {row['intent']} ({row['sentiment']})"):
                    st.write("**Customer Message:**")
                    st.write(row['customer_message'])
                    st.write("**Agent Response:**")
                    st.write(row['agent_response'])
                    st.write(f"**Timestamp:** {row['timestamp']}")
        else:
            st.info("No conversations found. Please generate sample data first.")
            if st.button("Generate Sample Data"):
                st.info("Please run: `python scripts/generate_data.py`")
                
    except Exception as e:
        st.error(f"Error loading conversations: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Contact Center AI v1.0**")
st.sidebar.markdown("Built with Streamlit, FastAPI, and AI/ML")