import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('automobile.csv')

st.markdown("<h1 style = 'color: #2E3020; text-align: center; font-size: 60px; font-family: Georgia'>AUTOMOBILE PRICE PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #B30000; text-align: center; font-family: italic'>Built By EME ITA</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('NEWFLEET.png', width = 600, caption ='Built by Eme Ita')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>BACKGROUND OF STUDY</h2>", unsafe_allow_html = True)


st.markdown("<p>The primary objective of this machine learning project is to develop an accurate and robust predictive model for estimating the price of a car based on its various features. By leveraging advanced machine learning algorithms, the aim is to create a model that can analyze and learn from historical car data, encompassing attributes such as make, model, year, mileage, engine type, fuel efficiency, and other relevant parameters</p>", unsafe_allow_html = True)

st.sidebar.image('automobile_user-removebg-preview.png')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader('User Input Variables')

sel_cols = ['curb-weight', 'normalized-losses', 'symboling', 'body-style', 'make', 'height',
            'city-mpg', 'horsepower', 'price']

curb_weight = st.sidebar.number_input('curb-weight', data['curb-weight'].min(), data['curb-weight'].max())
norm_loses = st.sidebar.number_input('normalized-losses', data['normalized-losses'].min(), data['normalized-losses'].max())
make = st.sidebar.selectbox('make', data['make'].unique())
body_style = st.sidebar.selectbox('body-style', data['body-style'].unique())
horse_power = st.sidebar.selectbox('horsepower', data['horsepower'].unique())
city_mpg = st.sidebar.number_input('city-mpg', data['city-mpg'].min(), data['city-mpg'].max())
height = st.sidebar.number_input('height', data['height'].min(), data['height'].max())
length = st.sidebar.number_input('length', data['length'].min(), data['length'].max())
price = st.sidebar.number_input('price', data['price'].min(), data['price'].max())

#users input
input_var = pd.DataFrame()
input_var['curb-weight'] = [curb_weight]
input_var['normalized-losses'] = [norm_loses]
input_var['make'] = [make]
input_var['body-style'] = [body_style]
input_var['horsepower'] = [horse_power]
input_var['city-mpg'] = [city_mpg]
input_var['height'] = [height]
input_var['length'] = [length]
input_var['price'] = [price]

st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

# import the transformers
body_style = joblib.load('body-style_encoder.pkl')
horse_power = joblib.load('horsepower_encoder.pkl')
make = joblib.load('make_encoder.pkl')

# transform the users input with the imported encoders
input_var['body-style'] = body_style.transform(input_var[['body-style']])
input_var['horsepower'] = horse_power.transform(input_var[['horsepower']])
input_var['make'] = make.transform(input_var[['make']])

st.header('Transformed Input Variable')
st.dataframe(input_var, use_container_width = True)


# st.dataframe(input_var)
model = joblib.load('AutomobileModel.pkl')

#to have a button for the user
if st.button('Predict Price'):
    predicted_price = model.predict(input_var)
    st.success(f"The Price of this Car is  {predicted_price[0].round()}")




# Define valid username and password
VALID_USERNAME = "godstreasure"
VALID_PASSWORD = "6172839405"

# Function to authenticate users
def authenticate(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD

# Streamlit app layout
def main():

    # Sidebar for login form
    st.sidebar.header("Login")
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")

    # Check if login button is clicked
    if st.sidebar.button("Login"):
        if authenticate(username_input, password_input):
            st.success("Logged in as {}".format(username_input))
            # You can proceed to show the main content of the app here
        else:
            st.error("Invalid username or password")

if __name__ == "__main__":
    main()
