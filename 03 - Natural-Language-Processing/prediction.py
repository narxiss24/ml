import joblib
import pandas as pd
import numpy as np

mdl = joblib.load('supply_model.pkl')

#%%
list = ['drg460 SCREW 3.5X30MM Bone-screw internal fixation system, non-sterile 150-299'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%%
list = ['drg460 SCREW 3.5X30MM Bone-screw internal fixation system, non-sterile 500-999'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%%
list = ['drg460 SCREW 3.5X30MM Bone-screw internal fixation system, non-sterile >=5000'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%%
list = ['drg470 SCREW 3.5X30MM Bone-screw internal fixation system, non-sterile 150-299'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%%
list = ['drg470 SCREW 3.5X30MM Bone-screw internal fixation system, non-sterile 4000-4999'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%% REAL CASES

#%% Current classification classifies this $1100 item as Bone Screw and Screw Washer

list = ['drg520 VIPER CORTICAL FIX SCREW 5.00MM X 50MM 1000-1499'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))

#%% Current classification classifies this as a Bone Screw and Screw Washer at 0.78 confidence
list = ['drg494 END CAP Bone nail end-cap, non-sterile 1000-1499'] 
ser = pd.Series(list)

prediction = mdl.predict(ser)
prediction_score = mdl.predict_proba(ser)

# df = pd.DataFrame(predictions)
print(prediction)
score = prediction_score
print(np.around(np.amax(prediction_score, axis=1),decimals=2))
