import pandas as pd

df = pd.read_csv("data/Training Dataset.csv")

insights = []

# 1. Credit history
credit_approval = df[df['Credit_History'] == 1.0]['Loan_Status'].value_counts(normalize=True)
if credit_approval.get('Y', 0) > 0.7:
    insights.append("Applicants with a good credit history (1.0) are likely to be approved.")

# 2. Low income
low_income = df[df['ApplicantIncome'] < 3000]
low_income_approval = low_income['Loan_Status'].value_counts(normalize=True)
if low_income_approval.get('Y', 0) < 0.5:
    insights.append("Applicants with income less than 3000 are less likely to be approved.")

# 3. Property area
for area in df['Property_Area'].unique():
    area_approval = df[df['Property_Area'] == area]['Loan_Status'].value_counts(normalize=True)
    if area_approval.get('Y', 0) > 0.7:
        insights.append(f"Applicants in the {area} area tend to be approved for loans.")

# 4. Education
grad_approval = df[df['Education'] == 'Graduate']['Loan_Status'].value_counts(normalize=True)
nongrad_approval = df[df['Education'] == 'Not Graduate']['Loan_Status'].value_counts(normalize=True)
if abs(grad_approval.get('Y', 0) - nongrad_approval.get('Y', 0)) < 0.1:
    insights.append("Education level (Graduate or Not Graduate) does not significantly affect loan approval.")

# 5. Self-employed
self_emp_approval = df[df['Self_Employed'] == 'Yes']['Loan_Status'].value_counts(normalize=True)
if self_emp_approval.get('Y', 0) < 0.6:
    insights.append("Self-employed applicants have a slightly lower chance of approval.")

# Save to file
with open("data/loan_insights.txt", "w") as f:
    for i in insights:
        f.write(i + "\n")

print("Saved insights to data/loan_insights.txt")
