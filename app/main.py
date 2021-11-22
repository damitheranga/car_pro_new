from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

modelpro = pickle.load(open('app/careerprogression_xgBoostModel.pkl', 'rb'))
modelSalPro = pickle.load(open('app/modelsalaryxgboostv8.pkl', 'rb'))
modelskill = pickle.load(open("app/skillxgboostmodelv5.pkl", "rb"))


@app.route('/')
def index():
    return 'Welcome to career predictor'




@app.route('/api/v1/cp/career_pro', methods=['POST'])
def car_pro():
    output = {'career_progression': '','salray_progression':''
              }
    int_features = [x for x in request.form.values()]
    print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame(
        [final], columns=['Gender', 'Age', 'Marital Status', 'Ethnicity', 'Home Town', 'A/L Stream', 'Subject_1', 'Result_1', 'Subject_2', 'Result_2', 'Subject_3', 'Result_3', 'Univesity/Degree Awarding Institute', 'Degree', 'Duration Of Degree', 'Graduation Year', 'GPA', 'Knowledge on programming concepts', 'Knowledge on programming languages', 'Knowledge on software engineering concepts','Knowledge on UI/UX interface', 'Fluent communication English', 'Problem Solving', 'Creativity', 'Self-learning', 'Management', 'Team Playing', 'Decision Making','Comitment','Front End Development','Back End Development','Full Stack Development','Mobile Application Development','UI/UX','clear understanding of career','check available career opportunities','managing career challenges','Adapt to new technology','Overall satisfactory rate of your career life'])


    predictioncarPro = modelpro.predict(data_unseen)
    predictionSalPro = modelSalPro.predict(data_unseen)
    output = {'career_progression': json.dumps(predictioncarPro[0].tolist()),'salray_progression':json.dumps(predictionSalPro[0].tolist()),
              }
    return output

@app.route('/api/v1/cp/skill_recommend', methods=['POST'])
def skill_predict():
    output = {'Skill Prediction': '',
              }
    int_features = [x for x in request.form.values()]
    print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame(
        [final], columns=['Gender', 'Age', 'Marital Status', 'Ethnicity','Home Town','A/L Stream','Subject_1','Result_1','Subject_2','Result_2','Subject_3','Result_3','Univesity/Degree Awarding Institute','Degree','Duration Of Degree','Graduation Year','GPA','Knowledge on programming concepts','Knowledge on programming languages','Knowledge on software engineering concepts','Knowledge on UI/UX interface','Fluent communication English','Problem Solving','Creativity','Self-learning','Management','Team Playing','Decision Making','Comitment','Front End Development','Back End Development','Full Stack Development','Mobile Application Development','UI/UX','clear understanding of career','check available career opportunities','managing career challenges','Adapt to new technology','Overall satisfactory rate of your career life','Associate Software Engineer No Of Years','Software Engineer No of Years','Senior Software Engineer No of years','Tech Lead No of Years', 'Architect No of Years'])


    predictionSkill= modelskill.predict(data_unseen)
    
 

    output = {'skill_recomndation': json.dumps(predictionSkill[0].tolist()),
              }
    return output



if __name__ == "__main__":
    app.run(debug=True)
