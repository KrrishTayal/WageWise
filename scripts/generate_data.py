import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


np.random.seed(42)

# NEW: Expanded Location Data Structure (Country Factor * City Factor)
location_data = {
    'USA': {
        'factor': 1.5,
        'cities': {
            'San Francisco': 1.9, 'New York': 1.8, 'Seattle': 1.4, 
            'Austin': 1.2, 'Atlanta': 1.0, 'Remote': 1.0
        }
    },
    'UK': {
        'factor': 1.1,
        'cities': {
            'London': 1.5, 'Manchester': 1.1, 'Edinburgh': 1.2, 
            'Birmingham': 1.0, 'Bristol': 1.1, 'Remote': 1.0
        }
    },
    'India': {
        'factor': 0.7,
        'cities': {
            'Bangalore': 1.4, 'Mumbai': 1.3, 'Delhi': 1.2, 
            'Chennai': 1.1, 'Hyderabad': 1.3, 'Remote': 1.0
        }
    },
    'Australia': {
        'factor': 1.3,
        'cities': {
            'Sydney': 1.6, 'Melbourne': 1.5, 'Brisbane': 1.2, 
            'Perth': 1.1, 'Adelaide': 1.0, 'Remote': 1.0
        }
    }
}

def generate_salary_dataset(size=10000):
    
    tech_jobs = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'UX Designer']
    finance_jobs = ['Financial Analyst', 'Accountant', 'Investment Banker', 'Risk Manager']
    healthcare_jobs = ['Doctor', 'Nurse', 'Pharmacist', 'Medical Technician']
    other_jobs = ['Teacher', 'Marketing Manager', 'HR Specialist', 'Sales Executive']
    
    all_jobs = tech_jobs + finance_jobs + healthcare_jobs + other_jobs
    educations = ['High School', 'Bachelor', 'Master', 'PhD']
    company_sizes = ['Startup (1-50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000+)']
    
    tech_skills = ['Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript']
    business_skills = ['Excel', 'PowerPoint', 'Financial Modeling', 'Project Management']
    soft_skills = ['Leadership', 'Communication', 'Teamwork']
    all_skills = tech_skills + business_skills + soft_skills
    
    
    data = []
    
    for _ in range(size):
       
        age = np.random.randint(22, 60)
        experience = np.random.randint(0, 40)
        education = np.random.choice(educations, p=[0.1, 0.5, 0.3, 0.1])
        job = np.random.choice(all_jobs)
        company_size = np.random.choice(company_sizes)
        
        # NEW: Select country and city based on new structure
        country = np.random.choice(list(location_data.keys()))
        country_info = location_data[country]
        location = np.random.choice(list(country_info['cities'].keys()))
        
        # Base Salary Calculation
        if job in tech_jobs:
            base = 60000 + experience * 2500 + np.random.randint(-5000, 10000)
        elif job in finance_jobs:
            base = 55000 + experience * 2000 + np.random.randint(-5000, 8000)
        elif job in healthcare_jobs:
            base = 50000 + experience * 3000 + np.random.randint(-5000, 15000)
        else:
            base = 45000 + experience * 1800 + np.random.randint(-5000, 5000)
        
        # Education Multiplier (same)
        if education == 'Bachelor':
            base *= 1.0
        elif education == 'Master':
            base *= 1.2
        elif education == 'PhD':
            base *= 1.4
        else:
            base *= 0.8
            
        # NEW: Location Multiplier (Country Factor * City Factor)
        base *= country_info['factor'] * country_info['cities'][location]
        
        # Company Size Multiplier (same)
        if company_size == 'Startup (1-50)':
            base *= 0.9
        elif company_size == 'Small (50-200)':
            base *= 1.0
        elif company_size == 'Medium (200-1000)':
            base *= 1.1
        else:
            base *= 1.3
            
        # Skill Bonus - Increased minimum value for 'else'
        num_skills = np.random.randint(5, 9)
        skills = np.random.choice(all_skills, num_skills, replace=False)
       
        skill_bonus = 0
        for skill in skills:
            if skill in ['Machine Learning', 'AWS', 'Financial Modeling']:
                skill_bonus += 5000 # Increased impact
            elif skill in ['Python', 'SQL', 'Project Management']:
                skill_bonus += 3500
            elif skill in ['Leadership', 'JavaScript']:
                skill_bonus += 2500
            else:
                skill_bonus += 1500 # Increased minimum bonus
                
        base += skill_bonus
        
        salary = int(base + np.random.randint(-5000, 5000))
        
        row = {
            'age': age,
            'experience': experience,
            'education': education,
            'job_title': job,
            'location': location,
            'country': country, # NEW: Added country to the output
            'company_size': company_size,
            'skills': ', '.join(skills),
            'salary': salary
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

df = generate_salary_dataset(10000)
df.to_csv('data/salary_data.csv', index=False)