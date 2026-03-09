import pandas as pd
import joblib
from flask import Flask, request, render_template

# Load trained model
model = joblib.load(r"C:\Users\osama jafer\Desktop\bank_churn\models\bank.joblib")

app = Flask(__name__)

def prepare_data(data):
    CreditScore = int(data.get('CreditScore'))
    Geography = data.get('Geography')
    Gender = data.get('Gender')
    Age = int(data.get('Age'))
    Tenure = int(data.get('Tenure'))
    Balance = float(data.get('Balance'))
    NumOfProducts = int(data.get('NumOfProducts'))
    HasCrCard = int(data.get('HasCrCard'))
    IsActiveMember = int(data.get('IsActiveMember'))
    EstimatedSalary = float(data.get('EstimatedSalary'))

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

    if pred == 1:
        result = "Churn"
    else:
        result = "Not Churn"

    return result


if __name__ == "__main__":
    app.run(debug=True)