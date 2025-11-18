import pandas as pd
import numpy as np
import random
import re

# --- Mock LLM for Chatbot Advice ---

def mock_llm_response(prompt: str, df: pd.DataFrame) -> str:
    """
    A mock function to simulate an LLM giving salary advice based on a prompt
    and the internal dataset (df).
    
    In a real application, this would be an API call to OpenAI, Gemini, etc.
    """
    prompt_lower = prompt.lower()
    
    # Simple keyword-based logic for giving tailored advice
    if "salary" in prompt_lower and any(j in prompt_lower for j in df['job_title'].unique()):
        
        job_title_match = next((j for j in df['job_title'].unique() if j.lower() in prompt_lower), None)
        
        if job_title_match:
            job_data = df[df['job_title'] == job_title_match]['salary']
            if not job_data.empty:
                avg = job_data.mean()
                median = job_data.median()
                top_location = df[df['job_title'] == job_title_match]['location'].mode()[0]
                
                return f"For a **{job_title_match}** role, our data suggests an average salary of **${avg:,.0f}** with a median of **${median:,.0f}**. To maximize your pay, you might consider roles in **{top_location}** or focusing on a **Large (1000+)** company size."
            
    elif "skill" in prompt_lower or "learn" in prompt_lower:
        
        # A simple analysis of top skills
        skill_df = df.copy()
        for skill in ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling', 'Project Management', 'Leadership']:
            skill_df[f'has_{skill}'] = skill_df['skills'].apply(lambda x: 1 if skill in x else 0)
        
        skill_salaries = {
            skill: skill_df[skill_df[f'has_{skill}'] == 1]['salary'].mean()
            for skill in ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 'Financial Modeling']
        }
        
        best_skill = max(skill_salaries, key=skill_salaries.get)
        avg_salary_no_skills = skill_df[[col for col in skill_df.columns if not col.startswith('has_')]]['salary'].mean()
        
        return f"Based on our data, acquiring **{best_skill}** seems to offer the biggest salary boost, with an average salary of **${skill_salaries[best_skill]:,.0f}** for professionals who have it, compared to **${avg_salary_no_skills:,.0f}** without specialized skills. Always combine technical skills with **Leadership** and **Project Management** for the best outcomes."

    elif "negotiate" in prompt_lower:
        return "To negotiate effectively, always do three things: 1) **Research your value** using a tool like WageWise. 2) **State a specific range** that is slightly higher than your target. 3) **Focus on your unique contributions** and not just your time."
    
    elif "trends" in prompt_lower or "market" in prompt_lower:
        return get_mock_market_trends()
        
    
    return f"I'm an AI Salary Advisor. I can help with general advice, but for a detailed answer, please ask about a specific **Job Title**, **Skills to Learn**, or **Negotiation** strategies. (Your query: '{prompt}')"

# --- Placeholder for Real-time Trends ---

def get_mock_market_trends() -> str:
    """
    A placeholder to simulate fetching real-time market data.
    
    In a real app, this would involve calling a data provider API 
    (e.g., job boards, economic data APIs) or a custom web scraper.
    """
    trends = [
        "**Market Trend Alert**: Demand for **DevOps Engineers** is currently surging in **New York** with a **15%** increase in job postings over the last quarter.",
        "**High-Growth Sector**: The **Healthcare** industry shows the highest salary growth potential (up to **18%** year-over-year) for roles requiring **Machine Learning** skills.",
        "**Company Size Insight**: **Startup (1-50)** companies are increasingly prioritizing **JavaScript** and **UX Design** skills, often offering equity to offset a lower base salary than large firms.",
        "**Global Insight**: While salaries in **San Francisco** remain the highest, the cost-of-living adjusted best-paying jobs are moving towards **Austin** and **Seattle**."
    ]
    
    return "## Global Salary Market Trends\n\n" + '\n\n'.join([f"* {t}" for t in random.sample(trends, 2)])


def analyze_market_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes job title and location demand from the *internal* dataset (df).
    In a real application, this should be complemented by external data.
    """
    demand = df.groupby(['job_title', 'location']).size().reset_index(name='Job Postings')
    demand_summary = demand.sort_values('Job Postings', ascending=False).head(10)
    
    # Calculate a simple 'Average Salary' to include with demand
    salary_data = df.groupby(['job_title', 'location'])['salary'].mean().reset_index(name='Average Salary')
    
    # Merge for a combined view
    demand_summary = demand_summary.merge(salary_data, on=['job_title', 'location'])
    demand_summary['Average Salary'] = demand_summary['Average Salary'].apply(lambda x: f'${x:,.0f}')
    
    return demand_summary