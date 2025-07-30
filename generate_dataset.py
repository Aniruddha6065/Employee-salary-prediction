import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

experience = np.random.randint(0, 21, n_samples)
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
education = np.random.choice(education_levels, n_samples)
job_roles = ['Software Engineer', 'Data Scientist', 'Manager', 'HR', 'Sales', 'Accountant']
job_role = np.random.choice(job_roles, n_samples)
locations = ['Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Delhi', 'Pune', 'Remote']
location = np.random.choice(locations, n_samples)
industries = ['Tech', 'Finance', 'Healthcare', 'Education', 'Retail']
industry = np.random.choice(industries, n_samples)

# Base salary by job role (in INR per year)
base_salary = {
    'Software Engineer': 600000,
    'Data Scientist': 700000,
    'Manager': 1000000,
    'HR': 500000,
    'Sales': 400000,
    'Accountant': 450000
}

# Education bonus (in INR)
education_bonus = {
    'High School': 0,
    'Bachelor': 50000,
    'Master': 100000,
    'PhD': 150000
}

# Location bonus (in INR)
location_bonus = {
    'Mumbai': 70000,
    'Bangalore': 80000,
    'Hyderabad': 40000,
    'Chennai': 50000,
    'Delhi': 60000,
    'Pune': 40000,
    'Remote': 0
}

# Industry bonus (in INR)
industry_bonus = {
    'Tech': 100000,
    'Finance': 80000,
    'Healthcare': 50000,
    'Education': -20000,
    'Retail': -30000
}

salary = []
for i in range(n_samples):
    s = base_salary[job_role[i]]
    s += education_bonus[education[i]]
    s += location_bonus[location[i]]
    s += industry_bonus[industry[i]]
    s += experience[i] * np.random.randint(20000, 50000)  # per year experience bonus
    s += np.random.normal(0, 30000)  # noise
    salary.append(int(s))

data = pd.DataFrame({
    'Experience (years)': experience,
    'Education Level': education,
    'Job Role': job_role,
    'Location': location,
    'Industry': industry,
    'Salary': salary
})

data.to_csv('employee_salary_data.csv', index=False)
print('Indian salary dataset generated and saved as employee_salary_data.csv')