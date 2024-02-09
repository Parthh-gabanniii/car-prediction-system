# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
#
# st.title('Car PreDictor SyStem')
#
#
# car_dict= pickle.load(open('car_dict.pkl','rb'))
# car = pd.DataFrame(car_dict)
# car_company_dict = pickle.load(open('car_company_dict.pkl','rb'))
# car1 = pd.DataFrame(car_company_dict)
# car_year_dict = pickle.load(open('car_year_dict.pkl','rb'))
# car2 = pd.DataFrame(car_year_dict)
# car_fuel_dict = pickle.load(open('car_fuel_dict.pkl','rb'))
# car3 = pd.DataFrame(car_fuel_dict)
# car_km_dict = pickle.load(open('car_km_dict.pkl','rb'))
# car4 = pd.DataFrame(car_km_dict)
#
#
# if 'selected_company' not in st.session_state:
#     st.session_state.selected_company = car['name'].unique()[0]

# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
#
# st.title('Car Predictor System')
#
# # Load data from pickle files
# car_dict = pickle.load(open('car_dict.pkl', 'rb'))
# car = pd.DataFrame(car_dict)
# car_company_dict = pickle.load(open('car_company_dict.pkl', 'rb'))
# car1 = pd.DataFrame(car_company_dict)
# car_year_dict = pickle.load(open('car_year_dict.pkl', 'rb'))
# car2 = pd.DataFrame(car_year_dict)
# car_fuel_dict = pickle.load(open('car_fuel_dict.pkl', 'rb'))
# car3 = pd.DataFrame(car_fuel_dict)
# car_km_dict = pickle.load(open('car_km_dict.pkl', 'rb'))
# car4 = pd.DataFrame(car_km_dict)
#
# # Dropdowns
# option = st.selectbox(
#     'Select the company',
#     car['name'].unique()
# )
#
# # Filter other options based on the selected company
# filtered_models = car1[car1['name'] == option]['company'].unique()
# filtered_years = car2['year'].unique()
# filtered_fuels = car3['fuel_type'].unique()
# # filtered_kms = car4[car['name'] == option]['kms_driven'].unique()
#
# # Dropdowns for other features
# option1 = st.selectbox('Select the model', filtered_models)
# option2 = st.selectbox('Select the year of purchase', filtered_years)
# option3 = st.selectbox('Select the fuel type', filtered_fuels)
# option4 = st.text_input('Enter the no. of km that the car has traveled')
#
# # option4 = st.selectbox('Enter the no. of km that the car has traveled', filtered_kms)
#
# if st.button('Predict Price'):
#     st.write(f"Selected Company: {option}")
#     st.write(f"Selected Model: {option1}")
#     st.write(f"Selected Year: {option2}")
#     st.write(f"Selected Fuel Type: {option3}")
#     # st.write(f"Selected KMs Driven: {option4}")
#     st.write(f"Entered KMs Driven: {option4}")





#
#
#
# option = st.selectbox(
# 'Select the company',
# car['name'].unique())
#
# option1 = st.selectbox(
# 'Select the model',
# car['company'].unique())
#
# option2 = st.selectbox(
# 'select the year of purchase',
# car['year'].unique())
#
# option3 = st.selectbox(
# 'Select the fuel type',
# car['fuel_type'].unique())
#
# option4 = st.selectbox(
# 'Enter the no. off km that the car has travelled ',
# car['kms_driven'].unique())
#
# if st.button('Predict Price'):
#     st.write(option)
#
# #
#

import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Car Predictor System')



# Load data from pickle files
car_dict = pickle.load(open('car_dict.pkl', 'rb'))
car = pd.DataFrame(car_dict)
car_company_dict = pickle.load(open('car_company_dict.pkl', 'rb'))
car1 = pd.DataFrame(car_company_dict)
car_year_dict = pickle.load(open('car_year_dict.pkl', 'rb'))
car2 = pd.DataFrame(car_year_dict)
car_fuel_dict = pickle.load(open('car_fuel_dict.pkl', 'rb'))
car3 = pd.DataFrame(car_fuel_dict)
car_km_dict = pickle.load(open('car_km_dict.pkl', 'rb'))
car4 = pd.DataFrame(car_km_dict)

cars = pickle.load(open('cars.pkl', 'rb'))

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.write(cars)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()








# Dropdowns
option = st.selectbox(
    'Select the company',
    car['name'].unique()
)

# Filter other options based on the selected company
filtered_models = car1[car1['name'] == option]['company'].unique()
filtered_years = car2[car2['name'] == option]['year'].unique()
filtered_fuels = car3[car3['name'] == option]['fuel_type'].unique()
filtered_kms = car4[car4['name'] == option]['kms_driven'].unique()

# Dropdowns for other features
option1 = st.selectbox('Select the model', filtered_models)
option2 = st.selectbox('Select the year of purchase', filtered_years)
option3 = st.selectbox('Select the fuel type', filtered_fuels)
option4 = st.selectbox('select no. of km that the car has traveled', filtered_kms)

lr.fit(cars.drop(columns={'Price','name','company','fuel_type'}), cars['Price'])
if st.button('Predict Price'):
    # Create a DataFrame with the selected options
    # input_data = pd.DataFrame([[option1, option, option2, option4, option3]],
    #                           columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Assuming 'pipe' is your pre-trained machine learning model

    # predicted_price = pipe.predict([[option, option1, option2, option4, option3]])
    predicted_price=lr.predict([[option2,option4]])


    st.write(f"Selected Company: {option}")
    st.write(f"Selected Model: {option1}")
    st.write(f"Selected Year: {option2}")
    st.write(f"Selected Fuel Type: {option3}")
    st.write(f"Selected KMs Driven: {option4}")
    st.write(f"Predicted Price: {predicted_price[0]}")


#

