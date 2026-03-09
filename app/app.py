import pandas as pd
import joblib
from flask import Flask, request, render_template



model = joblib.load(r'C:\Users\osama jafer\Desktop\bank_churn\models\bank.joblib')
app = Flask(__name__)

def prepare_data(data):
    CreditScore = data.get('CreditScore')
    Geography = data.get('Geography')
    Gender = data.get('Gender')
    Age = data.get('Age')
    Tenure = data.get('Tenure')
    Balance = data.get('Balance')
    NumOfProducts = data.get('NumOfProducts')
    HasCrCard = data.get('HasCrCard')
    IsActiveMember = data.get('IsActiveMember')
    EstimatedSalary = data.get('EstimatedSalary')
    Exited = data.get('Exited')
    example_customer = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary],
        "Exited": [Exited],
    })
    return example_customer


@app.route('/', methods=['GET'])
def home():
   return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    prepared_data = prepare_data(data)
    pred = model.predict(prepared_data)[0]
    if pred == 'Yes':
        return 'Churn'
    else:
        return 'Not Churn'

app.run()