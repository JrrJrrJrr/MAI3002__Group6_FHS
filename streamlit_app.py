import streamlit as st
import os
import base64

st.set_page_config(
    page_title="Covid Data Analysis",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)
st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")


# Helper: robust image display with fallbacks to placeholders
def show_sidebar_logo():
    # Try common logo files then fall back to placeholder in `placeholders/`
    for name in ("Ulogo.png", "Ulogo.svg", "Ulogo.jpg", "Ulogo.jpeg"):
        if os.path.exists(name):
            st.sidebar.image(name)
            return
    placeholder = os.path.join("placeholders", "Ulogo.svg")
    if os.path.exists(placeholder):
        st.sidebar.image(placeholder)
    else:
        st.sidebar.write("")


def show_image_any(base_path, caption=None, use_column_width=True):
    """Display image by trying several extensions and a placeholders/ fallback.
    `base_path` can be a path without extension (e.g. './covid') or a full filename.
    """
    exts = ["", ".png", ".svg", ".jpg", ".jpeg", ".gif"]
    # if a full filename was given and exists, show it directly
    if os.path.exists(base_path):
        st.image(base_path, caption=caption, use_column_width=use_column_width)
        return
    # try appending extensions
    for e in exts:
        p = base_path + e
        if os.path.exists(p):
            st.image(p, caption=caption, use_column_width=use_column_width)
            return
    # final fallback to placeholders/<basename>.svg
    fallback = os.path.join("placeholders", os.path.basename(base_path) + ".svg")
    if os.path.exists(fallback):
        st.image(fallback, caption=caption, use_column_width=use_column_width)
        return
    # give a minimal text fallback so app doesn't crash
    if caption:
        st.write(caption)
    else:
        st.write("")

# show the sidebar logo now
show_sidebar_logo()


############################# start page content #############################
st.title("Exploring the Facets of COVID-19: A Multi-Question Analysis")
st.write("✍️Authors: Amr Alshatnawi, Hailey Pangburn, Richard McMasters")
st.write("---")
st.header("MSBI 32000 Winter 2024 Semester - The University of Chicago")
st.write("""Welcome to our  COVID-19 data analysis web application, where we delve into the intricacies of the COVID-19 pandemic through data.
          Our mission is to uncover the hidden patterns, trends, and insights within the vast amounts of case surveillance data provided by the CDC.
          This project is a collaborative effort, aimed at enhancing our understanding of the pandemic's impact across different geographies and demographics.""")


st.markdown("""Here, you'll find a comprehensive analysis structured into various sections:
- **Introduction:** Get to know the background and objectives of our analysis.
- **Research Questions:** Explore the key questions guiding our investigation.
- **Data Source:** Learn about the CDC's COVID-19 Case Surveillance Public Use Data and how it's utilized in our study.
- **Data Exploration and Munging:** This section outlines our initial steps in preparing and understanding the dataset. We start by examining the structure, quality, and distribution of the COVID-19 Case Surveillance Public Use Data.
- **Analysis:** Dive into our findings with interactive visualizations and code snippets that bring the data to life.
- **Findings & Conclusion:** Discover the significant patterns and insights we've uncovered through our analysisand reflect on the implications of our findings and the potential paths forward.""")

st.info("""**Acknowledgement**  
This project was completed by students in the Intermediate Applied Data Analysis class within the Biomedical Informatics Program at the University of Chicago. Our work represents a collaborative educational endeavor, where we applied our learning to real-world data to contribute to the broader understanding of the COVID-19 pandemic.""")

# INTRODUCTION PAGE
import streamlit as st
import base64

st.set_page_config(
    page_title="Covid Data Analysis - Introduction",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")

show_sidebar_logo()

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################

st.title("Introduction")
st.divider()

st.markdown("""
The COVID-19 pandemic emerged as a catastrophic global event, touching every corner of the world.
This unprecedented crisis led to significant changes in our daily routines, leaving an indelible impact on our lives and
the way we work, even after its conclusion. To date, nearly **704 million** individuals have been impacted by COVID-19,
with the death toll surpassing **7 million**. Notably, the United States accounts for over a million of these fatalities,
as reported by the CDC. Through this analysis, our objective is to delve into the United States' COVID-19 data to uncover
insights regarding the pandemic's effects across various regions and demographic groups. We aim to determine if certain age
groups were more susceptible to infection and also explore whether specific variables can predict mortality rates.
""")

#st.image("./covid.png")



def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Display covid image with fallback placeholder
show_image_any("./covid", caption=None)

# RESEARCH QUESTION

import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Research Questions",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
    show_sidebar_logo()

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################


st.title("Research Questions")
st.divider()

st.subheader("1.Is there a significant difference in COVID-19 case counts between different age groups?")
st.markdown("""- **H0**: The distribution of COVID-19 cases across age groups is proportional to the population distribution of those age groups, indicating that age, relative to its population size, does not influence the likelihood of contracting COVID-19.
- **H1**: The distribution of COVID-19 cases across age groups is not proportional to the population distribution of those age groups, suggesting that, relative to their population size, certain age groups are more likely to contract COVID-19 than others.
""")

st.subheader("2. Do gender, age group, and case year significantly associate with COVID-19 mortality outcomes?")
st.markdown("""- **H0**: Gender, age group, and case year do not significantly predict COVID-19 mortality.
- **H1**: Gender, age group, and case year significantly predict COVID-19 mortality.""")

# DATA SOURCE

import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

st.set_page_config(
    page_title="Covid Data Analysis - Data Source",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()

############################# start page content #############################


st.title("Data Source and Description")
st.divider()

st.markdown("""
In our analysis, we utilized the CDC's *COVID-19 Case Surveillance Public Use Data with Geography* dataset,
which is regularly updated. The version we accessed was last updated on February 20, 2024.
This comprehensive dataset comprises approximately **105 million** rows, each representing a de-identified patient case.
It features **19 columns**, encompassing a wide range of information including demographic details, geographical data,
and related elements. Detailed descriptions of the columns are provided below.
""")

columns = {
    'Column Name': ['case_month','res_state', 'state_fips_code', 'res_county', 'county_fips_code', 'age_group', 'sex', 
                    'race','ethnicity', 'case_positive_specimen_interval', 'case_onset_interval', 'process', 'exposure_yn',
                    'current_status', 'symptom_status', 'hosp_yn', 'icu_yn','death_yn', 'underlying_conditions_yn'],
    'Description': ['The earlier of month the Clinical Date (date related to the illness or specimen collection) or the Date Received by CDC',
                    'State of residence','State FIPS code','County of residence','County FIPS code','Age group [0 - 17 years; 18 - 49 years; 50 - 64 years; 65 + years; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Sex [Female; Male; Other; Unknown; Missing; NA, if value suppressed for privacy protection.]','Race [American Indian/Alaska Native; Asian; Black; Multiple/Other; Native Hawaiian/Other Pacific Islander; White; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Ethnicity [Hispanic; Non-Hispanic; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Weeks between earliest date and date of first positive specimen collection','Weeks between earliest date and date of symptom onset.',
                    'Under what process was the case first identified? [Clinical evaluation; Routine surveillance; Contact tracing of case patient; Multiple; Other; Unknown; Missing]',
                    'In the 14 days prior to illness onset, did the patient have any of the following known exposures: domestic travel, international travel, cruise ship or vessel travel as a passenger or crew member, workplace, airport/airplane, adult congregate living facility (nursing, assisted living, or long-term care facility), school/university/childcare center, correctional facility, community event/mass gathering, animal with confirmed or suspected COVID-19, other exposure, contact with a known COVID-19 case? [Yes, Unknown, Missing]',
                    'What is the current status of this person? [Laboratory-confirmed case, Probable case]',
                    'What is the symptom status of this person? [Asymptomatic, Symptomatic, Unknown, Missing]',
                    'Was the patient hospitalized? [Yes, No, Unknown, Missing]','Was the patient admitted to an intensive care unit (ICU)? [Yes, No, Unknown, Missing]',
                    'Did the patient die as a result of this illness? [Yes; No; Unknown; Missing; NA, if value suppressed for privacy protection.]',
                    'Did the patient have one or more of the underlying medical conditions and risk behaviors: diabetes mellitus, hypertension, severe obesity (BMI>40), cardiovascular disease, chronic renal disease, chronic liver disease, chronic lung disease, other chronic diseases, immunosuppressive condition, autoimmune condition, current smoker, former smoker, substance abuse or misuse, disability, psychological/psychiatric, pregnancy, other. [Yes, No, blank]']
    
}

with st.expander("👆 Click to see columns description"):
    columns_df = pd.DataFrame(columns)
    st.table(columns_df)

st.write("")
st.write("")

URL_CDC = "https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data"

st.markdown(
    f'<a href="{URL_CDC}" style="display: inline-block; padding: 12px 20px; background-color: #085492; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">**Click here to access the dataset on CDC website**</a>',
    unsafe_allow_html=True
)

show_image_any("./cdc", caption=None)

# DATA EXPLORATION

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Covid Data Analysis - Data Exploration and Munging",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()


############################# start page content #############################

st.title("Data Exploration and Munging")
st.divider()


st.header("Sampling the Data")
st.markdown("""Given the extensive size of our dataset, containing 105 million records,
             we employed systematic sampling to generate a manageable sample dataset.
             This approach simplifies processing, analysis, and overall handling.
             We decided to select every 100th record from the original dataset for our sample.""")

with st.expander("👆 Expand to view systematic sampling code"):
    st.code("""
import pandas as pd

# data file path
file_path = 'drive/MyDrive/Data_Analysis/CDC_Covid_Data.csv'
# sample every 100th record
sampling_interval = 100  
chunk_size = 10000

# Placeholder for sampled rows
sampled_rows = []

# Open the file and iterate over it in chunks
with pd.read_csv(file_path, chunksize=chunk_size, na_values=['Missing', 'Unknown', 'NA', 'NaN']) as reader:
    for chunk_number, chunk in enumerate(reader):
        # Calculate the row index within the original file (global index) and select row
        start_row = chunk_number * chunk_size
        end_row = start_row + len(chunk)
        rows_to_sample = range(start_row, end_row, sampling_interval)

        # Adjust rows_to_sample to local indices within the chunk
        local_indices_to_sample = [row % chunk_size for row in rows_to_sample if row >= start_row and row < end_row]

        # Append the sampled rows from this chunk to the list
        sampled_rows.append(chunk.iloc[local_indices_to_sample])

# Concatenate all sampled rows into a single DataFrame
sampled_df = pd.concat(sampled_rows, ignore_index=True)

# file path to save the sampled DataFrame
output_file_path = 'systematic_sampled_covid_data_1M.csv'

# Save the DataFrame to a CSV file
sampled_df.to_csv(output_file_path, index=False)

            """)



st.header("Exploring the Data")

#data = pd.read_csv("systematic_sampled_covid_data.csv", na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])

# data 
csv_url = "https://www.dropbox.com/scl/fi/3ssnswv3158des6o0102v/systematic_sampled_covid_data_1M.csv?rlkey=kmf3ym19wdl3bq006fii5mcqa&st=8h53tkov&dl=1"
data = pd.read_csv(csv_url, na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])
st.write(f"The sampled dataset contains 1,045,441 rows and {data.shape[1]} columns.")
st.markdown("Data preview: ")
st.write(data.head())
with st.expander("👆 Expand to view code"):
    st.code("""
csv_url = "https://www.dropbox.com/scl/fi/3ssnswv3158des6o0102v/systematic_sampled_covid_data_1M.csv?rlkey=kmf3ym19wdl3bq006fii5mcqa&st=8h53tkov&dl=1"
data = pd.read_csv(csv_url, na_values=['Missing', 'Unknown', 'NA', 'NaN', '', ' '])
st.write(f"The sampled dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
print("Data preview: ")
data.head()
        """)

st.subheader("How have COVID-19 case counts varied over time since the onset of the pandemic?")


data['case_month'] = pd.to_datetime(data['case_month'], format='%Y-%m')

# Ensure dataframe is sorted by case_month
monthly_cases = data.groupby('case_month').size().reset_index(name='cases')
monthly_cases = monthly_cases.sort_values('case_month')

# Creating an interactive line plot with Plotly
fig = px.line(monthly_cases, x='case_month', y='cases', title='COVID-19 Cases Over Time',
              labels={'case_month': 'Month', 'cases': 'Number of Cases'},
              markers=True)

# Improve layout
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Number of Cases',
                  #width=900,
                  #height=700,
                  xaxis=dict(rangeslider=dict(visible=True), type='date')) 
st.plotly_chart(fig)
st.markdown("""The plot illustrates the overall trend in COVID-19 case numbers from the pandemic's onset,
            highlighting a significant peak at the beginning of 2022. 
           This surge in cases is likely attributable to the emergence of the omicron variant,
            which was identified in November 2021. """)

with st.expander("👆 Expand to view plot code"):
    st.code("""
import plotly.express as px

data['case_month'] = pd.to_datetime(data['case_month'], format='%Y-%m')

# Ensure dataframe is sorted by case_month
monthly_cases = data.groupby('case_month').size().reset_index(name='cases')
monthly_cases = monthly_cases.sort_values('case_month')

# Creating an interactive line plot with Plotly
fig = px.line(monthly_cases, x='case_month', y='cases', title='COVID-19 Cases Over Time',
              labels={'case_month': 'Month', 'cases': 'Number of Cases'},
              markers=True)

# Improve layout
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Number of Cases',
                  xaxis=dict(rangeslider=dict(visible=True), type='date')) 

fig.show()
""")
    

st.subheader("How do COVID-19 case counts vary among U.S. states?")
cases_per_state = data.groupby('res_state').size().reset_index(name='cases')


fig = px.choropleth(cases_per_state,
                    locations='res_state',
                    locationmode="USA-states",
                    color='cases',
                    scope="usa",
                    title='COVID-19 Cases by State',
                    hover_name='res_state',
                    hover_data={'res_state': False, 'cases': True}, 
                    color_continuous_scale=px.colors.sequential.YlOrRd, 
                    labels={'cases': 'Case Count'})  

# Enhance layout
fig.update_layout(
    title=dict(x=0.5),  
    geo=dict(
        lakecolor='rgb(255, 255, 255)', 
        showlakes=True,  # Show lakes
        landcolor='rgb(217, 217, 217)' 
    ),
    #width=900, 
    #height=700,
    margin=dict(t=50, l=0, r=0, b=0)
)

# Adjust color scale bar
fig.update_coloraxes(colorbar=dict(
    title='Total Cases',  
    thickness=20,  
    len=0.75, 
    bgcolor='rgba(255,255,255,0.5)',
    tickfont=dict(color='black'),  
    titlefont=dict(color='black')  
))


st.plotly_chart(fig)

st.markdown("""The choropleth map showcases the distribution of COVID-19 case counts, 
            highlighting that California has the highest number of cases,
             followed by Texas, New York, and Florida. This trend could be related to the
             larger population sizes in these states, considering they are the four most populated states in the U.S.""")

with st.expander("👆 Expand to view plot code"):
    st.code("""
import plotly.express as px

cases_per_state = data.groupby('res_state').size().reset_index(name='cases')


fig = px.choropleth(cases_per_state,
                    locations='res_state',
                    locationmode="USA-states",
                    color='cases',
                    scope="usa",
                    title='COVID-19 Cases by State',
                    hover_name='res_state',
                    hover_data={'res_state': False, 'cases': True}, 
                    color_continuous_scale=px.colors.sequential.YlOrRd, 
                    labels={'cases': 'Case Count'})  

# Enhance layout
fig.update_layout(
    title=dict(x=0.5),  
    geo=dict(
        lakecolor='rgb(255, 255, 255)', 
        showlakes=True,  # Show lakes
        landcolor='rgb(217, 217, 217)' 
    ),
    margin=dict(t=50, l=0, r=0, b=0)
)

# Adjust color scale bar
fig.update_coloraxes(colorbar=dict(
    title='Total Cases',  
    thickness=20,  
    len=0.75, 
    bgcolor='rgba(255,255,255,0.5)',
    tickfont=dict(color='black'),  
    titlefont=dict(color='black')  
))

fig.show()""")
        
st.subheader("How do COVID-19 case counts vary by gender?")

# Gender dataftame 
sex_counts = data['sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(sex_counts, x='Sex', y='Count', 
             title='Count of COVID-19 Cases by Sex',
             labels={'Count': 'Number of Cases'},
             color='Sex',
             color_discrete_map={'Female':'magenta', 'Male':'gold', 'Other':'green'})

st.plotly_chart(fig)

st.markdown("""The bar graph above illustrates the distribution of COVID-19 cases by sex, 
            revealing that **females** account for **544,048** cases, constituting **52%** of the total,
             while **males** represent **457,350** cases, making up **44%**. Additionally, there are **12** cases categorized as **'other'**,
             contributing to less than **1% closer to 0%** of the total, and **44,031** cases are marked as **missing** data,
             which comprises approximately **4%** of the overall case count.
""")

with st.expander("👆 Expand to view code"):
    st.code("""
import plotly.express as px

# Gender dataftame 
sex_counts = data['sex'].value_counts().reset_index()
sex_counts.columns = ['Sex', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(sex_counts, x='Sex', y='Count', 
             title='Count of COVID-19 Cases by Sex',
             labels={'Count': 'Number of Cases'},
             color='Sex',
             color_discrete_map={'Female':'pink', 'Male':'blue', 'Other':'green'})
fig.show()
""")


st.subheader("How do COVID-19 case counts vary by Race?")
# Race dataftame 
race_counts = data['race'].value_counts().reset_index()
race_counts.columns = ['Race', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(race_counts, x='Race', y='Count', 
             title='Count of COVID-19 Cases by Race',
             labels={'Count': 'Number of Cases'},
             color='Race',
             )
fig.update_layout(width=900, height=700)
st.plotly_chart(fig)

st.markdown("""The plot displayed above illustrates the distribution of COVID-19 cases by race,
indicating that the White population has a significantly higher number of cases relative to other racial groups.""")

with st.expander("👆 Expand to view code"):
    st.code("""
import plotly.express as px

# Race dataftame 
race_counts = data['race'].value_counts().reset_index()
race_counts.columns = ['Race', 'Count']  # Rename columns to match what we'll refer to in the plot

# Using the dataFrame in Plotly
fig = px.bar(race_counts, x='Race', y='Count', 
             title='Count of COVID-19 Cases by Race',
             labels={'Count': 'Number of Cases'},
             color='Race',
             )
fig.update_layout(width=900, height=700)
fig.show()
            
""")

# ------------------ Modeling: Predicting Mortality ------------------
st.header("Modeling: Predicting COVID-19 Mortality")
st.markdown("""Train a logistic regression model to predict whether a case resulted in death (`death_yn`).
You can limit rows used for training to speed up execution and optionally apply SMOTE to address imbalance.
""")

with st.expander("👆 Expand to view modeling options"):
    st.markdown("""
    - Choose the number of rows to use for training (sampling from the top of the dataframe).
    - Optionally apply SMOTE (requires `imbalanced-learn`).
    - Click **Train model** to run preprocessing, training and view evaluation metrics.
    """)

try:
    # UI controls
    max_rows = min(200000, data.shape[0])
    sample_n = st.number_input("Rows to use for modeling (approx)", min_value=1000, max_value=max_rows, value=50000, step=1000)
    test_size = st.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    use_smote = st.checkbox("Apply SMOTE to training set (requires imbalanced-learn)")
    train_button = st.button("Train model")

    if train_button:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

        # Prepare a reduced dataframe for modeling to keep runtime reasonable
        df_model = data.copy()

        # Keep only rows with clear death labels
        df_model = df_model[df_model['death_yn'].isin(['Yes', 'No'])]

        # Convert target to binary
        df_model = df_model[['sex', 'age_group', 'case_month', 'death_yn']].dropna(subset=['death_yn'])
        df_model['death_yn_bin'] = df_model['death_yn'].map({'Yes': 1, 'No': 0})

        # Convert case_month to year
        df_model['case_month'] = pd.to_datetime(df_model['case_month'], errors='coerce')
        df_model['case_year'] = df_model['case_month'].dt.year

        # Drop rows with missing predictor values
        df_model = df_model.dropna(subset=['sex', 'age_group', 'case_year'])

        # Limit rows to sample_n for performance
        df_model = df_model.head(int(sample_n))

        X = df_model[['sex', 'age_group', 'case_year']].copy()
        y = df_model['death_yn_bin'].astype(int)

        cat_features = ['sex', 'age_group']
        num_features = ['case_year']

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features),
            ('num', StandardScaler(), num_features)
        ], remainder='drop')

        if use_smote:
            # If SMOTE requested, do preprocessing then SMOTE, train sklearn classifier separately
            try:
                from imblearn.over_sampling import SMOTE
            except Exception:
                st.error("`imbalanced-learn` not installed. Install it with `pip install imbalanced-learn` to use SMOTE.")
                raise

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)

            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)

            clf = LogisticRegression(max_iter=200, class_weight='balanced')
            clf.fit(X_res, y_res)

            y_pred = clf.predict(X_test)
        else:
            # Use pipeline (preprocessing + classifier)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
            ])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Model Evaluation")
        st.write(f"Rows used for modeling: {df_model.shape[0]}")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall (sensitivity): {rec:.4f}")
        st.write(f"F1 score: {f1:.4f}")

        st.write("Confusion Matrix:")
        st.write(cm)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

except Exception as e:
    # If sklearn or other packages aren't installed, show a friendly message but do not crash the app
    st.error(f"Modeling section encountered an error: {e}")
    st.info("If this is due to missing packages, install them: `pip install scikit-learn imbalanced-learn` and restart the app.")

# FINDINGS 

import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - Findings & Conclusion",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()


############################# start page content #############################

st.title("Findings & Conclusion")
st.divider()



st.markdown("""

Our analysis aimed to uncover patterns in COVID-19 case counts and mortality outcomes, focusing on the roles of age, gender, and case year.
Initially, we explored the distribution of COVID-19 cases across various age groups. The data revealed a significant deviation in the 18 to 49 age group,
which displayed a disproportionately high number of cases. Utilizing the Chi-square goodness-of-fit test, we determined this variance to be statistically
significant, with a p-value less than 0.05. This finding suggests certain age groups, notably the 18 to 49 demographic, are more susceptible to contracting
COVID-19 relative to their population size, potentially due to factors like social behavior and employment types.

In examining COVID-19 mortality outcomes, logistic regression analysis highlighted that gender, age group, and case year are significant predictors of mortality,
with all predictors showing statistical significance (p-values < 0.05). Despite an initial dataset imbalance, our resampling strategy, which included both
undersampling and oversampling techniques, allowed us to maintain the model's overall significance while revealing an increased baseline probability of death
in a more balanced dataset context. This adjustment suggests a refined understanding of mortality risk factors. However, the model's precision at 10.62%
indicates a high rate of false positives, a challenge balanced by its strong sensitivity (81.44%) in accurately identifying actual deaths. This emphasizes
the model's utility in critical public health scenarios despite its need for further optimization to reduce false positives and improve the F1 score (0.1879).

In conclusion, our findings confirm the significant impact of gender, age, and case year on COVID-19 mortality, underscoring the importance of targeted public
health strategies. While the model presents areas for improvement, its ability to predict true positives remains a valuable asset in managing the pandemic
response, highlighting the potential for further refinement to enhance its predictive accuracy and applicability.


""")

# REFERENCES

import streamlit as st

st.set_page_config(
    page_title="Covid Data Analysis - References",
    page_icon="📊",
    #layout="wide"   
)
#st.markdown("""<style>body {zoom: 1.4;  /* Adjust this value as needed */}</style>""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size: 17px;">✍️ <strong>Authors:</strong></div> 
\n&nbsp;                                  
<a href="https://www.linkedin.com/in/amralshatnawi/" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Amr Alshatnawi&nbsp;&nbsp;</a><br>             

<a href="https://www.linkedin.com/in/hailey-pangburn" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">&nbsp;&nbsp;Hailey Pangburn&nbsp;&nbsp;</a><br>             
                    
<a href="mailto:mcmasters@uchicago.edu" style="display: inline-block; padding: 5px 7px; background-color: #871212; color: white; text-align: center; text-decoration: none; font-size: 15px; border-radius: 4px;">Richard McMasters</a><br>
""", unsafe_allow_html=True)

st.sidebar.write("---")
st.sidebar.markdown("""📅 March 9th, 2024""")
show_sidebar_logo()

# def add_side_title():
#     st.markdown(
#         """
#         <style>
#             [data-testid="stSidebarNav"]::before {
#                 content:"MSBI 32000 Winter 2024";
#                 margin-left: 20px;
#                 margin-top: 20px;
#                 font-size: 25px;
#                 position: relative;
#                 top: 80px;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# add_side_title()

############################# start page content #############################

st.title("References")
st.divider()

st.markdown("""
1. Brownlee, J. (2021, March 16). Smote for imbalanced classification with python. MachineLearningMastery.com. https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
            
2. Centers for Disease Control and Prevention. (n.d.-a). About covid-19. Centers for Disease Control and Prevention. https://www.cdc.gov/coronavirus/2019-ncov/your-health/about-covid-19.html#:~:text=COVID%2D19%20(coronavirus%20disease%202019,%2C%20the%20flu%2C%20or%20pneumonia. 
            
3. Centers for Disease Control and Prevention. (n.d.-b). Covid-19 case surveillance public use data with geography. Centers for Disease Control and Prevention. https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4/about_data 
            
4. Coronavirus cases:. Worldometer. (n.d.). https://www.worldometers.info/coronavirus/ 
            
5. Randomundersampler#. RandomUnderSampler - Version 0.12.0. (n.d.). https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html 
            
6. United States population by age - 2023 united states age demographics. Neilsberg. (n.d.). https://www.neilsberg.com/insights/united-states-population-by-age/#pop-by-age 
""")