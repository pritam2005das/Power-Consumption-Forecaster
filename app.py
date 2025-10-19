# import necessary libraries
import streamlit as st
from huggingface_hub import hf_hub_download
import pandas as pd
import cloudpickle
from PIL import Image

# load model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id= "pritam2005/Power-Consumption-Forecaster",
        filename= "model.pkl"
    )
    with open(model_path, "rb") as f:
        return cloudpickle.load(f)

pipeline = load_model()
model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocessor']

# Streamlit
st.title("Power Consumption Forecaster")

# user input
st.image(Image.open("upload_format.png"))
file = st.file_uploader("upload a csv file, consists the data from January 1 to December 30 in the format shown in the picture.\nDate format: YYYY-MM-DD HH-MM-SS", type= ["csv"])
if st.button("predict"):
    if file is not None:
        df = pd.read_csv(file)
        if(df['Datetime'].dtypes != 'datetime64[ns]'):
            df['Datetime'] = pd.to_datetime(df['Datetime'], format= '%Y-%m-%d %H:%M:%S')
        df.set_index('Datetime', inplace= True)
        df = df.resample('D').mean()
        date_range = pd.date_range(start= df.index[-1], periods= df.shape[0] + 1, freq= 'D')
        df = preprocessor.fit_transform(df.drop(columns= 'PowerConsumption_Zone3'))
        prediction = model.get_forecast(steps= df.shape[0], exog= df)
        forecast = prediction.predicted_mean
        conf_interval = prediction.conf_int()

        # create dataframe for prediction
        df_predict = pd.DataFrame({
            'Date' : date_range[1:],
            'PowerConsumption_Zone3' : forecast,
            'Min_PowerConsumption_Zone3' : conf_interval.iloc[:,0],
            'Max_PowerConsumption_Zone3' : conf_interval.iloc[:,1]
        })
        df_predict.set_index(df_predict['Date'], inplace= True)
        df_predict.drop(columns= ['Date'], inplace= True)
        st.dataframe(df_predict)

        # plot dataframe
        st.line_chart(df_predict)

        # download dataframe
        st.download_button(
            label= "Download CSV",
            data= df_predict.to_csv(),
            file_name= "Forecast.csv",
            mime= "text/csv"
        )

    else:
        st.error("Error! Enter a valid CSV file!!")