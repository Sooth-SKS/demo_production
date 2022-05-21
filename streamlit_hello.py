import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter





st.set_page_config(layout="wide")

# dashboard title
st.title(" Clinical risk alert using Machine Learning")
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
    ('Acute bronchitis','Asthma', 'Anemia','Viral sinusitis','Chronic congestive heart failure'))  


if option == 'Acute bronchitis':
     df = pd.read_csv("acute_bronchitis_prediction.csv")
     
elif option == 'Asthma': 
     df = pd.read_csv("asthma_prediction.csv")
         
elif option == 'Anemia':     
    df = pd.read_csv("anemia_prediction.csv")
    
elif option == 'Viral sinusitis': 
     df = pd.read_csv("viral_sinusitis_prediction.csv")
         
elif option == 'Chronic congestive heart failure':     
    df = pd.read_csv("chronic_congestive_heart_failure_prediction.csv")
    
patient_filter = st.sidebar.selectbox("patient id", pd.unique(df["PATIENT"]))


patient_filter = st.sidebar.selectbox(
    "patient_id_common",
    ('05ce1ed1-df78-451c-aa13-5912ad75cd8a',
 '09911b10-bafe-4d92-8762-b5778bd53aba',
 '09a55dbb-d1db-48a2-a6e1-8d9b168a9d1d',
 '0ac0182b-f424-4a2c-a1cb-65fb4036fbb9',
 '0082e62a-b695-463d-bca9-4b4da77a21af',
 '009e7163-9422-4706-a38f-dd6c323108ca',
 '0143acdb-2aaa-41e1-8f4a-e899b77e0e60',
 '080cd686-24fa-4864-a60b-e11cc52111e8',
 '073618d0-e5b7-451f-8aaa-9217000901df',
 '06048af1-6d9f-4e30-91dc-9a039c1a2fd0',
 '05af9427-0744-4fb3-9a99-929b6ec96bb7',
 '09d139f0-08a4-446b-967d-dd7dc482ebd6',
 '03c20045-2d5f-4c0c-a0e3-eb8682f4350b',
 '0a936f35-7ffc-452c-8814-ac511d8b3087'))  
    
    
df['Prediction'] = round(df['Prediction'], 2)
df1 = df[(df["Prediction"]>= 0.9)]
df2 = df[(df["Prediction"]>= 0.7) & (df["Prediction"]<0.9)]
df_each = df.loc[df["PATIENT"] == patient_filter].sort_values('START_YEAR',ascending = True)
df1_each = df1.loc[df1["PATIENT"] == patient_filter]
df2_each = df2.loc[df2["PATIENT"] == patient_filter]


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
        
        
st.markdown("<hr/>", unsafe_allow_html=True)


with st.expander("Patient demographic details"):
     st.markdown('**Gender**: ' + str(df_each.iloc[-1]['GENDER']))
     st.markdown('**Race**: ' + str(df_each.iloc[-1]['RACE']))
     #st.markdown('**Number of visits**: ' + str(df_each.shape[0]))
     st.markdown('**year**@First visit: ' + str(df_each.iloc[0]['START_YEAR']))
     st.markdown('**year**@Last visit: ' + str(df_each.iloc[-1]['START_YEAR']))
     st.markdown('**Age**@Last record: ' + str(round(df_each.iloc[-1]['patient_age']))+' '+'yr')
     st.markdown('**Zip**@Last record: ' + str(df_each.iloc[-1]['ZIP']))

 
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
   label="Number of visits",
    value=df_each.shape[0])

kpi2.metric(
   label="High-risk visits",
   value=df1_each.shape[0])

kpi3.metric(
    label="Moderate-risk visits",
    value=df2_each.shape[0])

st.markdown("<hr/>", unsafe_allow_html=True)


     
#create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("<h4 style='text-align: center; color: black;'>Risk score for different visits </h4>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df_each['START_YEAR'],df_each["Prediction"],s=5)
    ax.set_xlabel("Visit year",fontsize= 6)
    ax.set_ylabel("Risk score",fontsize= 6)
    ax.xaxis.set_tick_params(labelsize=4)
    ax.yaxis.set_tick_params(labelsize=4)
    #ax.set_xlim([df_each.iloc[0]['START_YEAR'],df_each.iloc[-1]['START_YEAR']])
    ax.set_ylim([0,1])
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}')) # No decimal places
    
    
    x =  [df_each['START_YEAR'].values[0],df_each['START_YEAR'].values[df_each.shape[0]-1]]
    y1 = [0.9]
    y2 = [1.0]
    y3 = [0.7]
    y4 = [0.9]
   

    ax.fill_between(x, y1, y2, facecolor="red", color='red', alpha=0.15)          
    ax.fill_between(x, y3, y4,facecolor="orange",color='orange',alpha=0.15) 
    
    
    ax.text(df_each['START_YEAR'].values[df_each.shape[0]-2], 0.95, 'High-risk(>90%)',fontsize= 6)
    ax.text(df_each['START_YEAR'].values[df_each.shape[0]-2], 0.8, 'Moderate-risk(70-90%)',fontsize= 6)
    
    
    st.pyplot(fig)
    
   
with fig_col2:
    st.markdown("<h4 style='text-align: center; color: black;'>Encounter class for all the visits </h4>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(1,1,1)
    
    y = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().values
    mylabels = df_each.groupby('ENCOUNTERCLASS')['PATIENT'].count().index.values
    ax.pie(y, labels = mylabels,autopct = "%0.2f%%", textprops={'fontsize': 3}) 
    st.pyplot(fig)
    


st.markdown("<h6 style='text-align: left; color: black;'>Get the patient record of high-risk & moderate-risk visits </h6>", unsafe_allow_html=True)


with st.expander("High-risk visits"):
    df1_each_short = df1_each.drop(['PATIENT'], 1)
    st.dataframe(df1_each_short)
     
with st.expander("Moderate-risk visits"):
    df2_each_short = df2_each.drop(['PATIENT'], 1)
    st.dataframe(df2_each_short)
     


















