import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import random
import os
from dotenv import load_dotenv
import google.generativeai as genai 

st.set_page_config(page_title="AI Salary Predictor", layout="wide")

# Function to update the sidebar radio selection
def set_page(page_name):
    st.session_state['main_nav_radio'] = page_name

@st.cache_resource
def load_model():
    return joblib.load('models/salary_predictor.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/salary_data.csv')

@st.cache_data
def load_column_order():
    with open('models/column_order.json', 'r') as f:
        return json.load(f)

# Updated to include 'country'
def preprocess_input(input_dict, column_order):
    data = {col: 0 for col in column_order}
    # Added 'country' to features list
    for feat in ['age', 'experience', 'education', 'job_title', 'location', 'country', 'company_size']:
        if feat in input_dict and feat in column_order:
             data[feat] = input_dict[feat]
    for skill in input_dict['skills']:
        skill_col = f'skill_{skill.lower().replace(" ", "_")}'
        if skill_col in column_order:
            data[skill_col] = 1
    input_df = pd.DataFrame([data])[column_order]
    for col in column_order:
        if col.startswith('skill_'):
            input_df[col] = input_df[col].astype(int)
    return input_df

# Placeholder functions for new navigation items (as requested previously)
def render_real_time_check():
    st.header(" 🌎 Real-Time Salary Market Check")
    st.markdown("This uses **Google Search** to generate a query for up-to-date salary information.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        check_country = st.text_input("Country (e.g., USA)")
    with col2:
        check_job = st.text_input("Job Title (e.g., Data Scientist)")
    with col3:
        check_company = st.text_input("Company Name (Optional)")
        
    if st.button("Generate Market Query"):
        if check_country and check_job:
            search_query = f"Average {check_job} salary in {check_country}"
            if check_company:
                search_query += f" at {check_company}"
            
            st.subheader("Live Data Search Query")
            st.info("Copy the query below and paste it into Google to find live market data.")
            st.code(search_query)
            
            sim_avg_salary = random.randint(70000, 150000)
            st.markdown(f"""
            <div class="result-box" style="border-color: #3498db; background-color: #ecf0f1;">
                <p class="big-font">Simulated Average Market Salary: ${sim_avg_salary:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Please enter a Country and Job Title.")

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = (
    "You are the 'WageWise AI Salary Advisor'. Your role is to provide smart, actionable, "
    "and professional salary advice, career planning tips, and negotiation strategies. "
    "Keep responses concise and encouraging. Use the user's provided salary data context "
    "to inform your advice, but always prioritize general, market-driven career strategy."
)

@st.cache_resource
def configure_gemini():
    """Configures the Gemini client and initializes the chat session."""
    if not API_KEY:
        return None
    try:
        genai.configure(api_key=API_KEY)
        # Using a fast, conversational model
        model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=SYSTEM_INSTRUCTION)
        # Start a new chat session to maintain conversation history
        return model.start_chat(history=[])
    except Exception as e:
        return None
    
def render_chatbot(df):
    st.header(" 🤖 AI Salary Advisor Chatbot")
    st.markdown("<a id='ai-assistant-chatbot'></a>", unsafe_allow_html=True)
    st.markdown("Ask the advisor about salary negotiation, market value for a specific job, or which skills offer the highest return.")
    
    chat_session = configure_gemini()
    
    if chat_session is None:
        st.error("AI Advisor is **offline**. Please install the necessary libraries (`pip install google-generativeai python-dotenv`) and set your `GEMINI_API_KEY` in a `.env` file to activate.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your salary or career..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Contextualize the prompt with the user's internal data for better advice
                context_data = df.to_string(index=False)[:500] 
                full_prompt = (
                    f"The user is using a salary prediction app based on their internal data with columns like: {', '.join(df.columns)}. "
                    f"A small snippet is:\n{context_data}\n"
                    f"User Query: {prompt}"
                )
                
                try:
                    response = chat_session.send_message(full_prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an API error. (Error: {e})"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})



def main():
    df = load_data()
    model = load_model()
    column_order = load_column_order()

    
    st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .result-box {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0px;
        background-color: #f0f8ff;
    }
    /* FIX: Changed font color to black for readability */
    .result-box p {
        color: black;
    }
    
        
    </style>
    """, unsafe_allow_html=True)

    


       

    st.title(" WageWise: AI-Powered Salary Predictor")
    st.markdown("Predict your salary based on job market trends")

    st.sidebar.title("Navigation")
    
    # Initialize session state for navigation if not present
    if 'main_nav_radio' not in st.session_state:
        st.session_state['main_nav_radio'] = "Salary Prediction"
        
    app_mode = st.sidebar.radio("Go to", 
        ["Salary Prediction", "Career Growth Projection", "Top Paying Roles", "Real-Time Salary Check", "AI Assistant (Chatbot)"], 
        key="main_nav_radio")

    if app_mode == "Salary Prediction":
        render_salary_prediction(df, model, column_order)
    elif app_mode == "Career Growth Projection":
        render_career_growth(df, model, column_order)
    elif app_mode == "Top Paying Roles":
        render_top_paying_roles(df)
    elif app_mode == "Real-Time Salary Check":
        render_real_time_check()
    elif app_mode == "AI Assistant (Chatbot)":
        render_chatbot(df)


def render_salary_prediction(df, model, column_order):
    st.header(" AI-Powered Salary Estimation")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.selectbox("Job Title", sorted(df['job_title'].unique()))
        
        country = st.selectbox("Country", sorted(df['country'].unique()))
        country_locations = df[df['country'] == country]['location'].unique()
        location = st.selectbox("Location (City)", sorted(country_locations))
        company_size = st.selectbox("Company Size", sorted(df['company_size'].unique()))
    with col2:
        experience = st.slider("Years of Experience", 0, 40, 5)
        education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
        age = st.slider("Age", 22, 60, 30)
    
    
    st.subheader("Skills")
    top_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling', 'Project Management', 'Leadership']
    skills = []
    cols_per_row = 4

    for row_start in range(0, len(top_skills), cols_per_row):
        row_skills = top_skills[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_skills))
        for col, skill in zip(cols, row_skills):
            with col:
                if st.checkbox(skill, key=f"skill_{skill}"):
                    skills.append(skill)


    if st.button("Predict Salary"):
        input_data = {
            'age': age,
            'experience': experience,
            'education': education,
            'job_title': job_title,
            'location': location,
            'country': country, # Added country
            'company_size': company_size,
            'skills': skills
        }

        try:
            input_df = preprocess_input(input_data, column_order)
            prediction = model.predict(input_df)[0]

            st.markdown(f"""
            <div class="result-box">
                <p class="big-font">Predicted Salary: ${prediction:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            confidence = max(0.85, min(0.98, 0.9 + (experience / 100)))
            lower_bound = int(prediction * (1 - (1 - confidence)/2))
            upper_bound = int(prediction * (1 + (1 - confidence)/2))

            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Salary Prediction with Confidence"},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [lower_bound * 0.9, upper_bound * 1.1]},
                    'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': prediction},
                    'steps': [
                        {'range': [lower_bound * 0.9, lower_bound], 'color': "lightgray"},
                        {'range': [lower_bound, upper_bound], 'color': "lightgreen"},
                        {'range': [upper_bound, upper_bound * 1.1], 'color': "lightgray"}
                    ],
                    'bar': {'color': "darkblue"}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            peers = df[
                (df['job_title'] == job_title) &
                (df['experience'].between(experience-2, experience+2)) &
                (df['education'] == education) &
                (df['country'] == country) # Filter by country too
            ]

            if not peers.empty:
                avg_salary = peers['salary'].mean()
                percentile = np.mean(prediction > peers['salary']) * 100

                st.subheader("Comparison with Peers")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average for Similar Profiles", f"${avg_salary:,.0f}")
                    st.metric("Your Percentile", f"{percentile:.1f}%")
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=peers['salary'], name='Peers', boxpoints='all', jitter=0.3, pointpos=-1.8))
                    fig.add_trace(go.Scatter(x=['Peers'], y=[prediction], mode='markers', marker=dict(color='red', size=12), name='Your Prediction'))
                    fig.update_layout(title='Salary Distribution Comparison', yaxis_title='Salary')
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            
def render_career_growth(df, model, column_order):
    st.header(" Career Growth & Salary Projection")

    col1, col2 = st.columns(2)
    with col1:
        current_job = st.selectbox("Current Job Title", sorted(df['job_title'].unique()), key="current_job")
        current_exp = st.slider("Current Experience (years)", 0, 40, 5, key="current_exp")
        current_edu = st.selectbox("Current Education Level", ['High School', 'Bachelor', 'Master', 'PhD'], key="current_edu")
        # Added country for growth projection
        current_country = st.selectbox("Country of Employment", sorted(df['country'].unique()), key="current_country")
    with col2:
        future_job = st.selectbox("Future Job Title (promotion target)", sorted(df['job_title'].unique()), key="future_job")
        years_to_project = st.slider("Years to Project", 1, 15, 5, key="years_proj")
        future_edu = st.selectbox("Future Education Goal", ['Same as Current', 'Bachelor', 'Master', 'PhD'], key="future_edu")
        # Added location for growth projection
        country_locations = df[df['country'] == current_country]['location'].unique()
        current_location = st.selectbox("Current Location (City)", sorted(country_locations), key="current_location")

    st.subheader("Future Skills to Acquire")
    top_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling', 'Project Management', 'Leadership']
    future_skills = []
    cols_per_row = 4

    for row_start in range(0, len(top_skills), cols_per_row):
        row_skills = top_skills[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_skills))
        for col, skill in zip(cols, row_skills):
            with col:
                if st.checkbox(skill, key=f"future_skill_{skill}"):
                    future_skills.append(skill)


    if st.button("Generate Projection", key="proj_button"):
        years = list(range(current_exp, current_exp + years_to_project + 1))
        salaries, skill_progress = [], []

        for i, year in enumerate(years):
            progress = i / len(years)
            current_skills = []
            if progress > 0.5:
                num_skills_to_add = int(len(future_skills) * min(1, (progress - 0.5) * 2))
                current_skills = future_skills[:num_skills_to_add]

            input_data = {
                'age': 30 + (year - current_exp),
                'experience': year,
                'education': current_edu if future_edu == 'Same as Current' else future_edu,
                'job_title': current_job if progress < 0.5 else future_job,
                'location': current_location, # Used selected location
                'country': current_country, # Added country
                'company_size': 'Medium (200-1000)',
                'skills': current_skills
            }
            input_df = preprocess_input(input_data, column_order)
            salaries.append(model.predict(input_df)[0])
            skill_progress.append(", ".join(current_skills) if current_skills else "None")

        projection_df = pd.DataFrame({
            'Year': [2023 + (year - current_exp) for year in years],
            'Experience': years,
            'Salary': salaries,
            'New Skills': skill_progress
        })

        fig1 = px.line(projection_df, x='Year', y='Salary', title="Salary Projection Over Time", markers=True, hover_data=['New Skills'])
        fig1.update_traces(line_color='#4CAF50', line_width=3)
        fig1.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Skill Acquisition Timeline")
        fig2 = px.bar(projection_df, x='Year', y=[1]*len(projection_df), color='New Skills', title="When New Skills Will Be Added", labels={'New Skills': 'Skills Added'})
        fig2.update_layout(showlegend=True, yaxis_visible=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Key Milestones")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Starting Salary", f"${salaries[0]:,.0f}")
            st.metric("1 Year Growth", f"${salaries[1] - salaries[0]:,.0f}", delta=f"{((salaries[1]/salaries[0])-1)*100:.1f}%")
        with col2:
            st.metric(f"After {years_to_project} Years", f"${salaries[-1]:,.0f}")
            st.metric("Total Growth", f"${salaries[-1] - salaries[0]:,.0f}", delta=f"{((salaries[-1]/salaries[0])-1)*100:.1f}%")


def render_top_paying_roles(df):
    st.header(" Top Paying Roles")

    # Group by Country, then Job Title
    avg_salary_by_country_job = df.groupby(['country', 'job_title'])['salary'].mean().sort_values(ascending=False).reset_index()
    
    selected_country = st.selectbox("Filter by Country", ['All'] + sorted(df['country'].unique().tolist()))
    
    if selected_country != 'All':
        display_df = avg_salary_by_country_job[avg_salary_by_country_job['country'] == selected_country].head(10)
        chart_title = f"Top 10 Highest Paying Job Titles in {selected_country}"
    else:
        # Group by job title only for "All" view
        display_df = df.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(10).reset_index()
        display_df.rename(columns={'salary': 'salary_avg'}, inplace=True)
        chart_title = "Top 10 Highest Paying Job Titles (Global Avg)"
    
    fig = px.bar(
        display_df, 
        x=display_df.columns[-1], 
        y='job_title', 
        orientation='h', 
        title=chart_title, 
        labels={display_df.columns[-1]: 'Average Salary', 'job_title': 'Job Title'}, 
        color=display_df.columns[-1], 
        color_continuous_scale='greens'
    )
    fig.update_layout(xaxis_tickprefix='$')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Salary by Company Size")
    fig = px.box(df, x='company_size', y='salary', color='company_size', title="Salary Distribution by Company Size")
    fig.update_layout(yaxis_tickprefix='$')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Ensure session state is initialized before running main, though Streamlit handles this well
    if 'main_nav_radio' not in st.session_state:
        st.session_state['main_nav_radio'] = "Salary Prediction"
    main()
