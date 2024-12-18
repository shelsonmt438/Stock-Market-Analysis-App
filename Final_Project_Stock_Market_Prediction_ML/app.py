import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime  # Import the datetime module

model = load_model('DP_Bdat_Stock_Predictions_Model.keras')

st.markdown(
    """
    <style>
	.stApp {
        background-color: #ffc40c;
        max-width: 1200px;
        margin: 0 auto;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1 {
        color: black;
    }
    p {
        color: black;
    }
    h3{
        color: #536878;
    }
</style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header"><h1>BDAT - Stock Market Predictor</h1></div>', unsafe_allow_html=True)

# Hero banner
hero_banner = Image.open("logo.jpg")  
st.image(hero_banner, use_column_width=True)

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2024-12-21'

# Print the current date and time
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.markdown(f"**Current Date and Time:** {current_time}")

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
