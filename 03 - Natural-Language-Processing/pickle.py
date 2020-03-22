import joblib
import pandas as pd
import numpy as np

mdl = joblib.load('supply_model.pkl')

list = ['Bone-screw internal spinal fixation system, sterile'] 
ser = pd.Series(list) 

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))