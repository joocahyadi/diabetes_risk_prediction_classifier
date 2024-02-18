# Import libraries
import sys

import streamlit as st
import requests
import json

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

from src.exception import CustomException


# Main part of the script
if __name__ == "__main__":

    # UI part
    # Title
    st.title('Diabetes Prediction Web Application')

    # Introduction
    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader('Hello and welcome!')
    st.write('This web application will help people to determine whether they have a high chance of diabetes or not.')
    
    # Features
    ## Age
    age_option = st.select_slider(
        "Please input your age",
        options=[i for i in range(1, 101)]
    )

    ## Sex
    sex_option = st.selectbox(
        "Please input your gender", ("Male","Female"), index=None
    )

    ## Polyuria
    polyuria_option = st.selectbox(
        "Do you have polyuria?", ("Yes","No"),index=None
    )

    ## Polydipsia
    polydipsia_option = st.selectbox(
        "Do you have polydipsia?", ("Yes","No"), index=None
    )

    ## Sudden weight loss
    swl_option = st.selectbox(
        "Do you feel sudden loss in your weight?", ("Yes","No"), index=None
    )

    ## Weakness
    weakness_option = st.selectbox(
        "Do you feel any kind of weakness?", ("Yes","No"), index=None
    )

    ## Polyphagia
    polyphagia_option = st.selectbox(
        "Do you feel a strong and extreme hunger, leading to overeating?", ("Yes","No"), index=None
    )

    ## Genital thrush
    gt_option = st.selectbox(
        "Do you sense any presence of genital thrush?", ("Yes","No"), index=None
    ) 

    # Visual blurring
    vb_option = st.selectbox(
        "Do you feel that your visual is getting blurry?", ("Yes","No"), index=None
    )

    ## Itching
    itching_option = st.selectbox(
        "Do you feel any kind of itching lately?", ("Yes","No"), index=None
    )

    ## Irritability
    irritability_option = st.selectbox(
        "Do you feel any irritability lately?", ("Yes","No"), index=None
    )

    ## Delayed healing
    dh_option = st.selectbox(
        "Do you feel that your wound is hard to be healed?", ("Yes","No"), index=None
    )

    ## Partial paresis
    pp_option = st.selectbox(
        "Do you feel any partial loss of voluntary movement?", ("Yes","No"), index=None
    )

    ## Muscle stiffness
    ms_option = st.selectbox(
        "Do you feel any stiffness in your muscles?", ("Yes","No"), index=None
    )

    ## Alopecia
    alopecia_option = st.selectbox(
        "Do you feel any irregular hair loss?", ("Yes","No"), index=None
    )

    ## Obesity
    obesity_option = st.selectbox(
        "Do you feel any presence of obesity?", ("Yes","No"), index=None
    ) 

    # Sidebar
    with st.sidebar:
        # Give title
        st.title('Check Out The Links Below for Other Resources! :smile:')

        # Some links
        st.link_button(label='Analysis Notebook', url='https://github.com/joocahyadi/diabetes_risk_prediction_classifier/blob/main/notebook/diabetes_notebook.ipynb')
        st.link_button('Github Repository', url='https://github.com/joocahyadi/diabetes_risk_prediction_classifier/')

    # Send the input to the ML model in backend
    if st.button('Submit'):
        
        try:
        # Send the user's answers to the ML model and get the prediction result
            data = CustomData(age_option,sex_option,polyuria_option,polydipsia_option,swl_option,weakness_option,
                            polyphagia_option, gt_option,vb_option,itching_option,irritability_option,dh_option,
                            pp_option,ms_option,alopecia_option,obesity_option)
            df_data = data.convert_into_dataframe()

            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(df_data)
        
        except Exception as e:
            raise CustomException(e, sys.exc_info())


        # Print the prediction result to the screen
        if pred == 'Yes':
            st.subheader(":heavy_exclamation_mark: You're :red[having] a high risk of diabetes. Becareful!")
        else:
            st.subheader(":star2: You're not having a high risk of diabetes. Keep it up! :100:")