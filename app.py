from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('salary_predictor.joblib')

# Options for dropdowns
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
job_roles = ['Software Engineer', 'Data Scientist', 'Manager', 'HR', 'Sales', 'Accountant']
locations = ['Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Delhi', 'Pune', 'Remote']
industries = ['Tech', 'Finance', 'Healthcare', 'Education', 'Retail']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        experience = int(request.form['experience'])
        education = request.form['education']
        job_role = request.form['job_role']
        location = request.form['location']
        industry = request.form['industry']
        
        input_df = pd.DataFrame({
            'Experience (years)': [experience],
            'Education Level': [education],
            'Job Role': [job_role],
            'Location': [location],
            'Industry': [industry]
        })
        prediction = int(model.predict(input_df)[0])
    return render_template('index.html', 
                           prediction=prediction, 
                           education_levels=education_levels, 
                           job_roles=job_roles, 
                           locations=locations, 
                           industries=industries)

if __name__ == '__main__':
    app.run(debug=True) 