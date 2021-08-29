from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Loading pickle files of all the models:
model_diabetes = pickle.load(open('Pickle_Diabetes.pkl', 'rb'))
model_heart = pickle.load(open('Pickle_Heart.pkl', 'rb'))
model_kidney = pickle.load(open('Pickle_Kidney.pkl', 'rb'))
model_liver = pickle.load(open('Pickle_Liver.pkl', 'rb'))
model_breast = pickle.load(open('Pickle_Breast.pkl', 'rb'))
model_stroke = pickle.load(open('Pickle_Stroke.pkl', 'rb'))
model_medicalinsurancecost = pickle.load(open('Pickle_MedicalInsuranceCost.pkl', 'rb'))

# Home Page
@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')

# Diabetes
@app.route("/predict_diabetes", methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        values_diabetes = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]])
        prediction_diabetes = model_diabetes.predict(values_diabetes)

        return render_template('result_diabetes.html', prediction=prediction_diabetes)


# Heart
@app.route("/predict_heart", methods=['POST'])
def predict_heart():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        values_heart = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        prediction_heart = model_heart.predict(values_heart)

        return render_template('result_heart.html', prediction=prediction_heart)


# Kidney
@app.route("/predict_kidney", methods=['POST'])
def predict_kidney():
    if request.method == 'POST':
        sg = float(request.form['sg'])
        htn = float(request.form['htn'])
        hemo = float(request.form['hemo'])
        dm = float(request.form['dm'])
        al = float(request.form['al'])
        appet = float(request.form['appet'])
        rc = float(request.form['rc'])
        pc = float(request.form['pc'])

        values_kidney = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
        prediction_kidney = model_kidney.predict(values_kidney)

        return render_template('result_kidney.html', prediction=prediction_kidney)


# Liver
@app.route("/predict_liver", methods=['POST'])
def predict_liver():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Gender = float(request.form['Gender'])
        Total_Bilirubin = float(request.form['Total_Bilirubin'])
        Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens = float(request.form['Total_Protiens'])
        Albumin = float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])


        values_liver = np.array([[Age,Gender,Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
        prediction_liver = model_liver.predict(values_liver)

        return render_template('result_liver.html', prediction=prediction_liver)


# Breast Cancer
@app.route("/predict_breast", methods=['POST'])
def predict_breast():
    if request.method == 'POST':
        texture_mean = float(request.form['texture_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        texture_se = float(request.form['texture_se'])
        smoothness_se = float(request.form['smoothness_se'])
        symmetry_se = float(request.form['symmetry_se'])
        symmetry_worst = float(request.form['symmetry_worst'])

        values_breast = np.array([[texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean,
                            texture_se, smoothness_se, symmetry_se, symmetry_worst]])
        prediction_breast = model_breast.predict(values_breast)

        return render_template('result_breast.html', prediction=prediction_breast)


# Stroke
@app.route("/predict_stroke", methods=['POST'])
def predict_stroke():
    if request.method == 'POST':
        gender = request.form['gender']
        if gender == 'Male':
            gender_Male = 1
            gender_Female = 0
        else:
            gender_Male = 0
            gender_Female = 1

        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])


        work_type = request.form['work_type']

        if work_type == 'Never_worked':
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 0

        if work_type == 'Private':
            work_type_Never_worked = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 0

        elif work_type == "Self_employed":
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
            work_type_Govt_job = 0

        elif work_type == "children":
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
            work_type_Govt_job = 0

        else:
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 1


        smoking_status = request.form['smoking_status']

        if smoking_status == "formerly_smoked":
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 0
            smoking_status_Unknown = 0

        elif smoking_status == "never_smoked":
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_Smokes = 0
            smoking_status_Unknown = 0

        elif smoking_status == "Smokes":
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 1
            smoking_status_Unknown = 0

        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 0
            smoking_status_Unknown = 1


        values_stroke = np.array([[gender_Male,age, hypertension, heart_disease, ever_married,
                            Residence_type, avg_glucose_level, bmi,
                            work_type_Never_worked, work_type_Private,work_type_Self_employed, work_type_children,
                            smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_Smokes]])
        prediction_stroke = model_stroke.predict(values_stroke)

        return render_template('result_stroke.html', prediction=prediction_stroke)


# Medical Cost Insurance
@app.route("/predict_insurance", methods=['POST'])
def predict_insurance():
    if request.method == 'POST':
        age = float(request.form['age'])

        sex = request.form['sex']
        if (sex == 'male'):
            sex_male = 1
            sex_female = 0
        else:
            sex_male = 0
            sex_female = 1

        smoker = request.form['smoker']
        if (smoker == 'yes'):
            smoker_yes = 1
            smoker_no = 0
        else:
            smoker_yes = 0
            smoker_no = 1

        bmi = float(request.form['bmi'])
        children = int(request.form['children'])

        region = request.form['region']
        if (region == 'northwest'):
            region_northwest = 1
            region_southeast = 0
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southeast'):
            region_northwest = 0
            region_southeast = 1
            region_southwest = 0
            region_northeast = 0
        elif (region == 'southwest'):
            region_northwest = 0
            region_southeast = 0
            region_southwest = 1
            region_northeast = 0
        else:
            region_northwest = 0
            region_southeast = 0
            region_southwest = 0
            region_northeast = 1


        values_insurance = np.array([[age,sex_male,smoker_yes,bmi,children,region_northwest,region_southeast,region_southwest]])
        prediction_insurance = model_insurance.predict(values_insurance)
        prediction_insurance = round(prediction_insurance[0],2)


        return render_template('result_insurance.html', prediction_text='Estimate medical insurance cost is {}'.format(prediction_insurance))




if __name__ == "__main__":
    app.run(debug=True)

