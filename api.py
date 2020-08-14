from flask import Flask,render_template,request
import requests,os,pickle,jsonify,sklearn
import numpy as np

app=Flask(__name__)


current_path = os.getcwd()
pickle_path = os.path.join(current_path, "assets", "loan_predict.pkl")
model = pickle.load(open(pickle_path, "rb"))

@app.route("/")
@app.route("/home",methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method=='POST':
        loan_amnt=int(request.form['loan_amnt'])
        term=int(request.form['term'])
        int_rate=float(request.form['int_rate'])
        emp_length=float(request.form['emp_length'])
        home_ownership=int(request.form['home_ownership'])
        annual_inc=float(request.form['annual_inc'])
        annual_inc=np.log(annual_inc)
        purpose=int(request.form['purpose'])
        addr_state=int(request.form['addr_state'])
        dti=float(request.form['dti'])
        delinq_2yrs=float(request.form['delinq_2yrs'])
        revol_util=float(request.form['revol_util'])
        total_acc=float(request.form['total_acc'])
        longest_credit_length=float(request.form['longest_credit_length'])
        verification_status=int(request.form['verification_status'])
        prediction=model.predict([[loan_amnt,term,int_rate,emp_length,home_ownership,annual_inc,purpose,addr_state,dti,delinq_2yrs,revol_util,total_acc,longest_credit_length,verification_status]])
        output=prediction[0]
        if output==0:
            return render_template('index.html',prediction_text="It leads to Good Customer")
        else:
            return render_template('index.html',prediction_text="It leads to Bad Customer")
    else:
        return render_template("index.html")



@app.route("/api/predict",methods=["POST"])
def predict_api():
    if request.method=='POST':
        data = request.get_json()
        loan_amnt=int(data.form['loan_amnt'])
        term=int(data.form['term'])
        int_rate=float(data.form['int_rate'])
        emp_length=float(data.form['emp_length'])
        home_ownership=int(data.form['home_ownership'])
        annual_inc=float(data.form['annual_inc'])
        annual_inc=np.log(annual_inc)
        purpose=int(data.form['purpose'])
        addr_state=int(data.form['addr_state'])
        dti=float(data.form['dti'])
        delinq_2yrs=float(data.form['delinq_2yrs'])
        revol_util=float(data.form['revol_util'])
        total_acc=float(data.form['total_acc'])
        longest_credit_length=float(data.form['longest_credit_length'])
        verification_status=int(data.form['verification_status'])
        prediction=model.predict([[loan_amnt,term,int_rate,emp_length,home_ownership,annual_inc,purpose,addr_state,dti,delinq_2yrs,revol_util,total_acc,longest_credit_length,verification_status]])
        output=prediction[0]
        if output==0:
            return jsonify("It is not a loan Defaulter")
        return jsonify("It is loan Defaulter")


if __name__=="__main__":
    app.run(debug=True)