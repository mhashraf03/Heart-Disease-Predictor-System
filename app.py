from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


app=Flask(__name__)
# instantiate object

#loading the different saved model for heart disease
#heart_predict=pickle.load(open('heart1.pkl', 'rb'))

heart_predict=pickle.load(open('heart3.pkl', 'rb')) #in this case random forest algorithm

@app.route('/') # instancing one page (homepage)
def home():
    return render_template("home.html")
# ^^ open home.html, then see that it extends layout.
# render home page.

@app.route('/about/') # instancing child page
def about():
    return render_template("about.html")

@app.route('/heartdisease/') # instancing child page
def heartdisease():
    return render_template("heart.html")



@app.route('/predictheartdisease/',methods=['POST']) 
def predictheartdisease():      #function to predict heart disease
    int_features=[x for x in request.form.values()]
    processed_feature_heart=[np.array(int_features,dtype=float)]
    prediction=heart_predict.predict(processed_feature_heart)
    if prediction[0]==1: 
        display_text="The person has Heart Disease"
    else:
        display_text="The person doesn't have Heart Disease"
    return render_template('heart.html',output_text="Result: {}".format(display_text))


if __name__=="__main__":
    app.run(debug=True)