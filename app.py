import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('insurance.csv')

# Encoding categorical features
df.replace({'sex': {'male': 0, 'female': 1},
            'smoker': {'yes': 0, 'no': 1},
            'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}},
           inplace=True)

x = df.drop('charges', axis=1)
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

gr = GradientBoostingRegressor()
gr.fit(x_train, y_train)
y_pred = gr.predict(x_test)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title("Medical Insurance Cost Prediction")
# Instructions for input format
st.markdown("**Instructions:**")
st.markdown("Enter the person's features separated by commas in the following order:")
st.markdown("`age, sex (male: 0, female: 1), bmi, children, smoker (yes:1/no:0), region (southeast: 0, southwest: 1, northeast: 2, northwest: 3)`")
st.markdown("For example: `30, 0, 25.0, 2, 1, 2`")


input_text = st.text_input("Enter person's features")

input_text_splited=input_text.split(",")

try:
    np_df=np.asarray(input_text_splited,dtype=float)
    input_df_reshape=np_df.reshape(1,-1)
    prediction=gr.predict(input_df_reshape)
    st.write('Medical Insurance is:',prediction[0])
except ValueError:
    st.write('please enter numerical values')   



