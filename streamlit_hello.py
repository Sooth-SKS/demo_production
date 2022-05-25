import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter



st.set_page_config(layout="wide")

st.title(" Clinical risk alert using Machine Learning(ML)")

st.markdown("<h6 style='text-align: left; color: black;'>A precision prevention system that identifies clinical and demographic covariates that drive the onset of chronic conditions. </h6>", unsafe_allow_html=True)
#st.markdown("<h6 style='text-align: left; color:black;'> An end-to-end machine learning workflow that used simulated data from an EHR generates a risk-score against each visit. An overall risk-score is finally generated for each patient</h6>", unsafe_allow_html=True)
st.write("An end-to-end machine learning workflow that used simulated data from an EHR and generates a risk-score against each visit. An overall risk-score is finally generated for each patient")
st.markdown("<hr/>", unsafe_allow_html=True)
   
st.sidebar.subheader("Select disease & patient")

option = st.sidebar.selectbox(
    "disease condition",
    ('Acute bronchitis','Chronic congestive heart failure','Anemia','Asthma'))  


if option == 'Acute bronchitis':
     df = pd.read_csv("acute_bronchitis_prediction.csv")
     patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('019319e2-e1e6-4691-a74f-11285ff10b81','015a6eb2-ba4a-46e7-a093-44aa1c923a95'))
     
     if patient_filter == '019319e2-e1e6-4691-a74f-11285ff10b81':
         start = 7572
         text = '_predicted risk-score is zero, data shows no acute bronchitis event in the next visit_'
         
 
    
     elif patient_filter == '015a6eb2-ba4a-46e7-a093-44aa1c923a95':
         start = 7556
         text = '_predicted risk-score is zero, data shows no acute bronchitis event in the next visit_'
         
         
elif option == 'Chronic congestive heart failure':     
    df = pd.read_csv("chronic_congestive_heart_failure_prediction.csv")
    patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('970692ab-1eb1-4eb1-a528-88dbaad13aa6','0372c553-9f75-4167-ac3f-bbf89e6bed4c'))
    
    if patient_filter == '970692ab-1eb1-4eb1-a528-88dbaad13aa6':
        start = 2350
        text = '_predicted risk-score is one, data shows chronic congestive heart failure in the next visit_'
        
    
    elif patient_filter == '0372c553-9f75-4167-ac3f-bbf89e6bed4c':
        start = 3640
        text = '_predicted risk-score is close to zero, data shows no chronic congestive heart failure in the next visit_'
    
elif option == 'Anemia':     
    df = pd.read_csv("anemia_prediction.csv")
    patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('c756400e-d9a4-4848-ae47-909a300173b0','f34991ad-c1bb-41db-8b89-5c5464ecb887'))
    if patient_filter == 'c756400e-d9a4-4848-ae47-909a300173b0':
        start = 1773
        text = '_predicted risk-score is one, data shows anemia in the next visit_'
 
    
    elif patient_filter == 'f34991ad-c1bb-41db-8b89-5c5464ecb887':
        start = 3731  
        text = '_predicted risk-score is one, data shows anemia in the next visit_'
     

elif option == 'Asthma': 
     df = pd.read_csv("asthma_prediction.csv")
     patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('11b99ded-308c-4f45-b8e6-4a53af85b7c5','7b16d62e-c24c-4453-a0e6-af2b506d2ab2'))
     if patient_filter == '11b99ded-308c-4f45-b8e6-4a53af85b7c5':
        start = 114
        text = '_predicted risk-score is 0.82 , data shows asthma in the next visit_'
 
    
     elif patient_filter == '7b16d62e-c24c-4453-a0e6-af2b506d2ab2':
        start = 747 
        text = '_predicted risk-score is 0.95, data shows asthma in the next visit_' 



df['Prediction'] = round(df['Prediction'], 2)
df_each = df.loc[df["PATIENT"] == patient_filter].sort_values('START_YEAR',ascending = True)
a = df_each['Prediction'].iloc[df_each.shape[0]-2:df_each.shape[0]].mean()
b = df_each['Prediction'].mean()
c = 0.8*a+0.2*b




kpi1, fig_col1= st.columns(2)

kpi1.metric(
    label="Overall risk-score",
    value=str(round(c*100))+'%',
    
 )
with fig_col1:

    if c < 0.5:
        st.info("Extremly safe patient - No intervention is needed")
    elif (c>=0.50) & (c<0.70):
        st.info("Safe patient - intervention is optional")
    elif (c>=0.70) & (c<0.90):
        st.warning("Moderate-Risk patient - slight intervention is needed")
    elif c>=0.90:
        st.error("High-risk patient - Immediate intervention is needed")
        
        
with st.expander("Patient demographic details"):
     st.markdown('**Gender**: ' + str(df_each.iloc[-1]['GENDER']))
     st.markdown('**Race**: ' + str(df_each.iloc[-1]['RACE']))
     #st.markdown('**Number of visits**: ' + str(df_each.shape[0]))
     st.markdown('**year**@First visit: ' + str(df_each.iloc[0]['START_YEAR']))
     st.markdown('**year**@Last visit: ' + str(df_each.iloc[-1]['START_YEAR']))
     st.markdown('**Age**@Last record: ' + str(round(df_each.iloc[-1]['patient_age']))+' '+'yr')
     st.markdown('**Zip**@Last record: ' + str(df_each.iloc[-1]['ZIP']))


st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("<h6 style='text-align: left; color: black;'>Sample test visit:  </h6>", unsafe_allow_html=True)

st.text('current visit')
 


df1 = df.drop(['PATIENT','RACE','GENDER','ZIP','label','Prediction'], 1)
df1.rename(columns = {'ENCOUNTERCLASS':'ENCOUNTER', 'START_YEAR':'YEAR','patient_age':'AGE','past_conditions':'PAST DIAGNOSIS','present_conditions':'CURRENT DIAGNOSIS'}, inplace = True)
st.dataframe(df1.iloc[start:start+1,:])   


if st.button('Predicted risk-score for the next visit'):
    result = df.iloc[start:start+1,8].values[0]
    st.write(result)
    st.text('real next visit')
    st.dataframe(df1.iloc[start+1:start+2,:].drop(['PAST DIAGNOSIS'], 1))
    
    st.markdown("<hr />", unsafe_allow_html=True)
    st.markdown(text)
    st.markdown("<hr />", unsafe_allow_html=True)
    

    
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Risk-score & Encounter type:</h6>", unsafe_allow_html=True)     


fig_col1, space_col, fig_col2 = st.columns([2,1,2])  
with fig_col1:
    st.text("Predicted risk-score vs visit year")
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df_each['START_YEAR'],df_each["Prediction"],c='black',s=10)
    
    ax.set_xlabel("Visit year",fontsize= 6)
    ax.set_ylabel("Risk score",fontsize= 6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    #ax.set_xlim([df_each.iloc[0]['START_YEAR'],df_each.iloc[-1]['START_YEAR']])
    ax.set_xlim(1980,2020)
    ax.set_ylim([-0.1,1.1])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}')) # No decimal places
    #if df_each.loc[df_each["Prediction"]>=0.9] 
    ax.hlines(0.9,1980,2020,color = 'black',linestyles='dashed',linewidth=1)
    st.pyplot(fig)
    

   
   
with fig_col2:
    #st.markdown("<h4 style='text-align: center; color: black;'>Encounter class for all the visits </h4>", unsafe_allow_html=True)
    st.text("Encounter class distribution")
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1,1,1)
    
    y = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().values
    mylabels = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().index.values
    ax.pie(y, labels = mylabels,autopct = "%0.2f%%", textprops={'fontsize': 3}) 
    st.pyplot(fig)



     

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Dataset & modeling approach:</h6>", unsafe_allow_html=True)     

with st.expander("Dataset description"):
    st.text('The dataset is generated by Synthea, which is a Synthetic Patient Population Simulator.')
    st.text('The patient’s encounter records and demographic information are used.')
    st.text('There are 391,999 records exist in the final table for 10,000 patients.')

with st.expander("ML modeling approach"):
    st.text("The under-sampling method is used to balance the positive and negative class.")
    st.text("LogisticRegression with elastic net penalization is used for the classification.")
    st.text("Searched a grid of hyperparameters using hyperopt to find the best parameters.")
    
with st.expander("Different disease condition in the dataset"):
     st.text("Chronic pain, Chronic intractable migraine without aura, Impacted molars, Viral sinusitis , Acute viral pharyngitis")
     st.text("Acute bronchitis , Normal pregnancy, Anemia , Sinusitis, Streptococcal sore throat, Acute bacterial sinusitis")
     st.text("Childhood asthma, Perennial allergic rhinitis with seasonal variation, Perennial allergic rhinitis, Hyperlipidemia")












