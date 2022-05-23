import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import plotly.express as px





st.set_page_config(layout="wide")

# dashboard title
st.title(" Clinical risk alert using Machine Learning(ML)")
st.markdown("<h6 style='text-align: left; color: black;'>ML model which predicts risk-score against each visit for certain disease condition. It then finally calculate weighted risk-score for patients:  </h6>", unsafe_allow_html=True)
#st.subheader("Machine Learning model for predicting risk-score of certain disease condition ")


st.markdown("<hr/>", unsafe_allow_html=True)
   
st.sidebar.subheader("Select disease & patient")

    




#st.sidebar.subheader("Select predicted risk threshold")

#cmd_high = st.sidebar.radio(
#     'for high-risk visit',
#     ( '90-98%', '>98%'))
#if cmd_high == '90-98%':
#    threshold_0 = 0.90
 #   threshold_1 = 0.97
#elif cmd_high == '>98%':
#    threshold_0 = 0.98
#    threshold_1 = 1.00

    
    
#cmd_low = st.sidebar.radio(
#     'for low-risk visit',
#     ( '10-25%', '25-50%'))
#if cmd_low == '10-25%':
#    threshold1_0 = 0.10
#    threshold1_1 = 0.24
#elif cmd_low == '25-50%':
#    threshold1_0 = 0.25
#    threshold1_1 = 0.50

   


option = st.sidebar.selectbox(
    "disease condition",
    ('Acute bronchitis','Chronic congestive heart failure','Asthma', 'Anemia'))  


if option == 'Acute bronchitis':
     df = pd.read_csv("acute_bronchitis_prediction.csv")
     patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('019319e2-e1e6-4691-a74f-11285ff10b81','015a6eb2-ba4a-46e7-a093-44aa1c923a95'))
     
     if patient_filter == '019319e2-e1e6-4691-a74f-11285ff10b81':
         start = 7572
 
    
     elif patient_filter == '015a6eb2-ba4a-46e7-a093-44aa1c923a95':
         start = 7556
         
elif option == 'Chronic congestive heart failure':     
    df = pd.read_csv("chronic_congestive_heart_failure_prediction.csv")
    patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('970692ab-1eb1-4eb1-a528-88dbaad13aa6','0372c553-9f75-4167-ac3f-bbf89e6bed4c'))
    if patient_filter == '970692ab-1eb1-4eb1-a528-88dbaad13aa6':
        start = 2350
 
    
    elif patient_filter == '0372c553-9f75-4167-ac3f-bbf89e6bed4c':
        start = 3640
    
    
     
elif option == 'Asthma': 
     df = pd.read_csv("asthma_prediction.csv")
     patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('970692ab-1eb1-4eb1-a528-88dbaad13aa6','0372c553-9f75-4167-ac3f-bbf89e6bed4c'))
         
elif option == 'Anemia':     
    df = pd.read_csv("anemia_prediction.csv")
    patient_filter = st.sidebar.selectbox(
            "patient id", 
            ('970692ab-1eb1-4eb1-a528-88dbaad13aa6','0372c553-9f75-4167-ac3f-bbf89e6bed4c'))
            

    
    


# patient_filter = st.sidebar.selectbox("patient id", pd.unique(df["PATIENT"]))



df['Prediction'] = round(df['Prediction'], 2)
#df1 = df[(df["Prediction"]>= 0.9)]
#df2 = df[(df["Prediction"]>= 0.7) & (df["Prediction"]<0.9)]
df_each = df.loc[df["PATIENT"] == patient_filter].sort_values('START_YEAR',ascending = True)
#df1_each = df1.loc[df1["PATIENT"] == patient_filter]
#df2_each = df2.loc[df2["PATIENT"] == patient_filter]


#st.markdown("<h6 style='text-align: left; color: black;'>Cumulative risk score based on last 3 visits:  </h6>", unsafe_allow_html=True)
a = df_each['Prediction'].iloc[df_each.shape[0]-3:df_each.shape[0]].mean()
#st.text(str(round(a*100))+'%')

#st.markdown("<h6 style='text-align: left; color: black;'>Cumulative risk score based on all the visits:  </h6>", unsafe_allow_html=True)
b = df_each['Prediction'].mean()
#st.text(str(round(b*100))+'%')

#st.markdown("<h6 style='text-align: left; color: black;'>Weighted risk score:  </h6>", unsafe_allow_html=True)
c = 0.8*a+0.2*b
#st.text(str(round(c*100))+'%')



kpi1, fig_col1= st.columns(2)

kpi1.metric(
    label="Weighted Risk score",
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
 
#kpi1, kpi2, kpi3 = st.columns(3)

#kpi1.metric(
#   label="Number of visits",
#    value=df_each.shape[0])

#kpi2.metric(
#   label="High-risk visits",
#   value=df1_each.shape[0])

#kpi3.metric(
#    label="Moderate-risk visits",
#    value=df2_each.shape[0])

#st.markdown("<hr/>", unsafe_allow_html=True)
 
 

#kpi2, kpi6 = st.columns(2)   

#kpi1.metric(
#   label="Visit year",
#    value=df.iloc[2350:2351,1].values[0]) 

#kpi2.metric(
#   label="Patient Age",
#    value=df.iloc[2350:2351,6].values[0]) 

#kpi3.metric(
#   label="Encounter type",
#    value=df.iloc[2350:2351,0].values[0]) 

#kpi4.metric(
#   label="Past diagnosis",
#   value=df.iloc[2350:2351,9].values[0]) 

#kpi5.metric(
#   label="Present diagnosis",
#    value=df.iloc[2350:2351,10].values[0]) 

#with kpi2:
  
#kpi6.metric(
#   label="Predicted risk-score",
#    value=df.iloc[2350:2351,8].values[0]) 





df1 = df.drop(['PATIENT','RACE','GENDER','ZIP','label','Prediction'], 1)
#row = pd.IndexSlice[df1.index[df1.index == 2350], 3:5])
#df1.style.applymap(df_style, subset=df1.iloc[2350:2351,:])
df1.rename(columns = {'ENCOUNTERCLASS':'ENCOUNTER', 'START_YEAR':'YEAR','patient_age':'AGE','past_conditions':'PAST DIAGNOSIS','present_conditions':'CURRENT DIAGNOSIS'}, inplace = True)
st.dataframe(df1.iloc[start:start+1,:])   


if st.button('Predicted risk-score for the next visit'):
    result = df.iloc[start:start+1,8].values[0]
    st.write(result)
    st.text('real next visit')
    st.dataframe(df1.iloc[start+1:start+2,:].drop(['PAST DIAGNOSIS'], 1))
    
#if st.button('What is the real next visit ?'):
#   st.dataframe(df1.iloc[2351:2352,:])
    
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Risk-score & Encounter type:</h6>", unsafe_allow_html=True)     


fig_col1, fig_col2 = st.columns(2)  
#with st.expander("Predicted risk-score vs visit year"):
with fig_col1:
    #st.markdown("<h4 style='text-align: center; color: black;'>Risk score for different visits </h4>", unsafe_allow_html=True)
    st.text("Predicted risk-score vs visit year")
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df_each['START_YEAR'],df_each["Prediction"],c='black',s=10)
    
    ax.set_xlabel("Visit year",fontsize= 6)
    ax.set_ylabel("Risk score",fontsize= 6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_xlim([df_each.iloc[0]['START_YEAR'],df_each.iloc[-1]['START_YEAR']])
    ax.set_ylim([-0.1,1.1])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}')) # No decimal places
    st.pyplot(fig)
    #fig = px.scatter(df_each['START_YEAR'],df_each["Prediction"])
    
    
   # x =  [df_each['START_YEAR'].values[0],df_each['START_YEAR'].values[df_each.shape[0]-1]]
   # y1 = [0.9]
   # y2 = [1.0]
   # y3 = [0.7]
   # y4 = [0.9]
   

    #ax.fill_between(x, y1, y2, facecolor="red", color='red', alpha=0.15)          
    #ax.fill_between(x, y3, y4,facecolor="orange",color='orange',alpha=0.15) 
    
    
    #ax.text(df_each['START_YEAR'].values[df_each.shape[0]-2], 0.95, 'High-risk(>90%)',fontsize= 6)
    #ax.text(df_each['START_YEAR'].values[df_each.shape[0]-2], 0.8, 'Moderate-risk(70-90%)',fontsize= 6)
    
    #st.plotly_chart(fig, use_container_width=True)
    

   
   
#with st.expander("Encounter class distribution"):
with fig_col2:
    #st.markdown("<h4 style='text-align: center; color: black;'>Encounter class for all the visits </h4>", unsafe_allow_html=True)
    st.text("Encounter class distribution")
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1,1,1)
    
    y = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().values
    mylabels = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().index.values
    ax.pie(y, labels = mylabels,autopct = "%0.2f%%", textprops={'fontsize': 3}) 
    st.pyplot(fig)



#st.markdown("<h6 style='text-align: left; color: black;'>Get the patient record of high-risk & moderate-risk visits </h6>", unsafe_allow_html=True)


#with st.expander("High-risk visits"):
#    df1_each_short = df1_each.drop(['PATIENT'], 1)
#    st.dataframe(df1_each_short)
     
#with st.expander("Moderate-risk visits"):
#    df2_each_short = df2_each.drop(['PATIENT'], 1)
#    st.dataframe(df2_each_short)
     

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Dataset & modeling approach:</h6>", unsafe_allow_html=True)     

with st.expander("Dataset description"):
    st.text('The dataset is generated by Synthea, which is a Synthetic Patient Population Simulator')
    st.text('The patient’s encounter records and demographic information are used from electronic health record (EHR)')

with st.expander("ML modeling approach"):
    st.text(1)













