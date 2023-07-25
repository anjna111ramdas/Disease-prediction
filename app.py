from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn

class DataType(BaseModel):
    Symptoms1: str
    Symptoms2: str
    Symptoms3: str
    Symptoms4: str
    Symptoms5: str

app = FastAPI()
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']


"""
Sample JSON Input:- 
{
    "NPPM": 1,
    "LoanStatus": "no loans taken/all loans paid back duly",
    "Objective": "New Car Purchase",
    "Amount": 50000,
    "Guarantee": "co-applicant",
    "Experience": "between 1 and 4 years",
    "M_Status": "male and divorced/seperated",
    "ExistingLoan": 0,
    "Age": 35,
    "CA_Balance": "no current account",
    "SA_Balance": "greater than 1000",
    "PI_Balance": 15000,
    "WorkAB": "Yes",
    "PhNum": 0,
    "Tenure": 3,
    "prop": "Real Estate",
    "JobTyp": "skilled employee / official",
    "HouseT": "own",
    "NOE": 2
}
"""
# {'disease': hepatitis b, 
# 'test': [array of tests], 
# 'medicines ' :[array of medicines ] 
# }
def Preprocessing(tr):
    t2 = []
    for idx in range(0,95):
        t2.append(0)
    
    for i in range(0,len(l1)):
        for j in tr:
            if l1[i]==j:
                t2[i]=1
    return t2

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(item: DataType):
    print("hi")
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    # df = Preprocessing(df)
    temp_li = []
    temp_li = list(df.iloc[0].values)
    # print(temp_li)
    input_m = Preprocessing(temp_li)
    ans = model.predict([input_m])
    ans = ans[0]
    h='no'
    ans_vl = ""
    for a in range(0,len(disease)):
        if(ans == a):
            # h='yes'
            ans_vl = disease[ans]
            # print(ans_vl)
            break
        else:
            ans_vl = "Sorry Couldn't Detect Any Diseace"
    return ans_vl


@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}



"""
{
    "Symptoms1" : "knee_pain",
    "Symptoms2" : "mild_fever",
    "Symptoms3" : "muscle_pain",
    "Symptoms4" : "internal_itching",
    "Symptoms5" : "malaise"
}
"""
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)

