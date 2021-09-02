from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('Breast.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
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

        values = np.array([[texture_mean, smoothness_mean, compactness_mean, symmetry_mean, fractal_dimension_mean,
                            texture_se, smoothness_se, symmetry_se, symmetry_worst]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

