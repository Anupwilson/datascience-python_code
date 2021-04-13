import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title('Model Deployment: random forest classifier')

st.sidebar.header('User Input Parameters')


def user_input_features():
    CALLERID = st.sidebar.number_input("Insert the callerid")
    OPENBY = st.sidebar.number_input("Insert the open by")
    LOC = st.sidebar.number_input("Insert the loc")
    CATEGORY = st.sidebar.number_input("Insert the category")
    data = {'CALLERID': CALLERID,
            'OPENBY': OPENBY,
            'LOC': LOC,
            'CATEGORY': CATEGORY}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

incident = pd.read_csv("final_data.csv")
incident = incident.dropna()

X = incident.loc[:, ['caller_id', 'open_by', 'loc', 'category']]
Y = incident.loc[:, 'i_impact']
clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('high impact' if prediction_proba[0][1] > 0.5 else 'not high impact')

st.subheader('Prediction Probability')
st.write(prediction_proba)