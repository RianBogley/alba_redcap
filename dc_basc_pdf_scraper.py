# %%
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
print("Importing main libraries...")

# Standard libraries:
import datetime
import getpass
import glob
import math
import os
import pickle
import re
import shutil
import sys
import time

# Third party libraries:
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.datasets
import nilearn.decoding as decoding
import nilearn.image as image
import nilearn.maskers as maskers
import nilearn.plotting as plotting
import numpy as np
import openpyxl
import pandas as pd
import pingouin as pg
import plotly as py
import progressbar
import ptitprince as pt
import requests
import scipy.stats as stats
import seaborn as sns
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tabula as tb
import zentables as zen
from joblib import Parallel, delayed
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from nilearn.image import get_data, resample_to_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map, show
from PyPDF2 import PdfReader
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  RidgeCV)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_log_error)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.graphics.api import abline_plot, interaction_plot
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable

# ---------------------------------------------------------------------------- #
# %%
# ---------------------------------------------------------------------------- #
#                            IMPORT CUSTOM LIBRARIES                           #
# ---------------------------------------------------------------------------- #
print("Importing custom libraries...")
# Import custom libraries:
# NOTE: To user our custom scripts. 
# First set the directory where you want to download the code:
code_dir = 'C:/Users/rbogley/code/github_tools/'
# 
# First ensure you have access to our GitHub, 
# then in your terminal, "cd" to the directory you want to store the scripts,
# and run the following commands:
"""
git clone https://github.com/RianBogley/alba-redcap-tools.git
"""
# Once complete, you can import the scripts using the following code:
lava_tools_path = code_dir + 'alba-lava-tools'
neuroimaging_tools_path = code_dir + 'alba-neuroimaging-tools'
redcap_tools_path = code_dir + 'alba-redcap-tools'
misc_tools_path = code_dir + 'alba-misc-tools'

# ---------------------------- IMPORT REDCAP TOOLS --------------------------- #
sys.path.append(redcap_tools_path)
from redcap_export_data import (export_data_redcap, export_redcap_file,
                                export_redcap_metadata,
                                export_single_case_redcap,
                                export_single_case_single_instrument_redcap)
from redcap_import_data import import_data_redcap

# ---------------------------------------------------------------------------- #
# %%
# ---------------------------------------------------------------------------- #
#                              ACCSES REDCAP DATA                              #
# ---------------------------------------------------------------------------- #
# Specify necessary details: REDCap API Token, PIDN, & Visit Number:
print("Input your REDCap API token to export data from REDCap.")
token = getpass.getpass("Enter REDCap API token:")
pidn = input("Enter PIDN to export:")
visit = input("Enter REDCap Visit Number to export (e.g. 'visit_one_arm_1'):")
# ---------------------------------------------------------------------------- #
# Set instrument of interest:
instrument = 'basc'
# Export the specified instrument for that case:
basc_df = export_single_case_single_instrument_redcap(token, pidn, instrument, visit)

# ---------------------------------------------------------------------------- #
# Check which BASC Version is specified in REDCap:
if basc_df['basc_version'].values[0] == 3:
    field = 'basc3_upload'
    print('BASC Version 3 Specified.')
elif basc_df['basc_version'].values[0] == 2:
    field = 'basc2_upload'
    print('BASC Version 2 Specified.')
else:
    print('ERROR: BASC Version not specified.')
    print('Please specify BASC Version in REDCap and try again.')
    sys.exit()

basc_version = basc_df['basc_version'].values[0]
# ---------------------------------------------------------------------------- #
# Check if the BASC PDF in basc_df is empty or NaN:
if basc_df[field].values[0] == '' or basc_df[field].values[0] == 'NaN':
    print(f'ERROR: No BASC-{basc_df["basc_version"].values[0]} PDF uploaded.')
    print(f'Please upload the BASC-{basc_df["basc_version"].values[0]} PDF to REDCap and try again.')
    sys.exit()
elif basc_df[field].values[0] != '':
    print(f'BASC-{basc_df["basc_version"].values[0]} PDF appears to be uploaded.')
    print(f'Extracting data from {basc_df[field].values[0]}...')

# ---------------------------------------------------------------------------- #
# Export the PDF from REDCap using provided API token and set the filepath:
basc_pdf = export_redcap_file(token, pidn, visit, field)
basc_pdf_filepath = basc_pdf.name

# %%
# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

# TODO: CHECK IF THE PDF IF SELF-REPORT

# TODO: ADD VALIDITY INDEX READER

# TODO: RE-ADD COMMENTS AND CONCERNS AND STRENGTHS READER

def find_page_number(pdf_reader, keywords):
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if any(keyword in text for keyword in keywords):
            return i + 1

def find_table_number(data, keywords):
    for i, table in enumerate(data):
        if all(keyword in table.to_string() for keyword in keywords):
            return i

def clean_table(table):
    raw_column = table.loc[:, table.apply(lambda x: x.str.contains('Raw').any())].columns[0]
    try:
        t_score_column = table.loc[:, table.apply(lambda x: x.str.contains('T').any())].columns[0]
    except IndexError:
        t_score_column = table.loc[:, table.apply(lambda x: x.str.contains('Score').any() and not x.str.contains('Raw').any())].columns[0]
    percentile_column = table.loc[:, table.apply(lambda x: x.str.contains('Rank').any())].columns[0]
    

    # Rename the Raw Score, T Score, and Percentile columns:
    table.rename(columns={table.columns[0]: 'Domain'}, inplace=True)
    table.rename(columns={raw_column: 'Raw Score'}, inplace=True)
    table.rename(columns={t_score_column: 'T Score'}, inplace=True)
    table.rename(columns={percentile_column: 'Percentile'}, inplace=True)

    table = table.iloc[:, :4]

    # If any rows have "Developmental Social Disorders" as the value in the first column, print ('Boop')
    if any(table['Domain'] == 'Developmental Social Disorders'):
        pass
    # If any rows have "Developmental Social" as the value in the first column, replace it with "Developmental Social Disorders":
    elif any(table['Domain'] == 'Developmental Social'):
        # Replace the 'Developmental Social' string with 'Developmental Social Disorders':
        table.iloc[:, 0] = table.iloc[:, 0].str.replace('Developmental Social', 'Developmental Social Disorders')
        # Replace the values in the "Developmental Social Disorders" row with the values from the next row:
        dev_social_disorders_index = table.index[table['Domain'] == 'Developmental Social Disorders'].tolist()[0]
        table.loc[dev_social_disorders_index, 'Raw Score':'Percentile'] = table.loc[dev_social_disorders_index + 1, 'Raw Score':'Percentile'].values
        
        # Remove the row called "Disorders"
        table = table[table['Domain'] != 'Disorders']

    # Remove any rows that have a NaN in the first column:
    table = table[table.iloc[:, 0].notna()]

    # Reset the index:
    table.reset_index(drop=True, inplace=True)

    return table

def save_table_values(df, table, variables_dict):
    for variable in variables_dict:
        # Save the raw score to the dataframe to the column that matches the variable name + "_raw":
        df.loc[(df['id_number'] == pidn) & (df['redcap_event_name'] == visit), variables_dict[variable] + '_raw'] = table.loc[table['Domain'] == variable, 'Raw Score'].values[0]
        # Save the t score to the dataframe to the column that matches the variable name + "_tscore":
        df.loc[(df['id_number'] == pidn) & (df['redcap_event_name'] == visit), variables_dict[variable] + '_tscore'] = table.loc[table['Domain'] == variable, 'T Score'].values[0]
        # Save the percentile to the dataframe to the column that matches the variable name + "_per":
        df.loc[(df['id_number'] == pidn) & (df['redcap_event_name'] == visit), variables_dict[variable] + '_per'] = table.loc[table['Domain'] == variable, 'Percentile'].values[0]
    return df

# ---------------------------------------------------------------------------- #
# %%
# ---------------------------------------------------------------------------- #
#                             READ IN THE PDF DATA                             #
# ---------------------------------------------------------------------------- #
with open(basc_pdf_filepath, 'rb') as file:
    pdf_reader = PdfReader(file)
    
    # Find the page numbers for the tables:
    composite_score_page = find_page_number(pdf_reader, ['Composite Score Summary'])
    scale_score_page = find_page_number(pdf_reader, ['Scale Score Summary'])
    content_scale_score_page = find_page_number(pdf_reader, ['CONTENT SCALE SCORE TABLE:','The information provided below is'])
    validity_index_page = find_page_number(pdf_reader, ['VALIDITY INDEX SUMMARY', 'Validity Index Summary'])

data = tb.read_pdf(basc_pdf_filepath, pages='all', multiple_tables=True)

# Find the table numbers for the tables on each page using find_table_number:
# COMPOSITE SCORE SUMMARY TABLE:
composite_score_variables = {'Externalizing Problems': 'basc_externalizing',
                                'Internalizing Problems': 'basc_internalizing',
                                'Behavioral Symptoms Index': 'basc_behavior',
                                'Adaptive Skills': 'basc_adaptiveskill'}
# Find the table:
composite_score_table = data[find_table_number(data, composite_score_variables.keys())]
# Clean the table:
composite_score_table = clean_table(composite_score_table)

print(composite_score_table)

# Save the table values to the dataframe:
basc_df = save_table_values(basc_df, composite_score_table, composite_score_variables)

# SCALE SCORE SUMMARY TABLE:
scale_score_variables = {'Hyperactivity': 'basc_hyperactivity',
                            'Aggression': 'basc_aggression',
                            'Conduct Problems': 'basc_conduct',
                            'Anxiety': 'basc_anxiety',
                            'Depression': 'basc_depression',
                            'Somatization': 'basc_somatization',
                            'Atypicality': 'basc_atypicality',
                            'Withdrawal': 'basc_withdrawal',
                            'Attention Problems': 'basc_attention',
                            'Adaptability': 'basc_adaptability',
                            'Social Skills': 'basc_socialskill',
                            'Leadership': 'basc_leadership',
                            'Activities of Daily Living': 'basc_dailyliving',
                            'Functional Communication': 'basc_functcomm'}
# Find the table:
scale_score_table = data[find_table_number(data, scale_score_variables.keys())]
# Clean the table:
scale_score_table = clean_table(scale_score_table)

print(scale_score_table)

# Save the table values to the dataframe:
basc_df = save_table_values(basc_df, scale_score_table, scale_score_variables)

# CONTENT SCALE SCORE TABLE:
content_scale_score_variables = {'Anger Control': 'basc_angercontrol',
                                    'Bullying': 'basc_bullying',
                                    'Developmental Social Disorders': 'basc_socialdisord',
                                    'Emotional Self-Control': 'basc_emotioncontrol',
                                    'Executive Functioning': 'basc_execfunc',
                                    'Negative Emotionality': 'basc_negemotion',
                                    'Resiliency': 'basc_resiliency'}
# Find the table:
content_scale_score_table = data[find_table_number(data,[key for key in content_scale_score_variables.keys() if key != 'Developmental Social Disorders'])]
# Clean the table:
content_scale_score_table = clean_table(content_scale_score_table)

print(content_scale_score_table)

# Save the table values to the dataframe:
basc_df = save_table_values(basc_df, content_scale_score_table, content_scale_score_variables)

# ---------------------------------------------------------------------------- #

# %%
# ---------------------------------------------------------------------------- #
#                     EXPORT DATA TO CSV FOR REDCAP IMPORT                     #
# ---------------------------------------------------------------------------- #
# Find parent directory of basc_filepath:
output_dir = os.path.dirname(basc_pdf_filepath)
# Export the basc_df to csv:
basc_df.to_csv(f'{output_dir}/{pidn}_{visit}_basc-{basc_version}.csv', index=False)

# %%
# ---------------------------------------------------------------------------- #
#                        IMPORT THE DATA BACK TO REDCAP                        #
# ---------------------------------------------------------------------------- #
# WARNING: THIS WILL OVERWRITE EXISTING DATA IN REDCAP

# Import the basc_df back to REDCap via API using import_data_redcap:
import_data_redcap(token, basc_df, data='raw', headers='raw', overwriteBehavior='normal')


# %%
# ---------------------------------------------------------------------------- #
#                                 FINISHING UP                                 #
# ---------------------------------------------------------------------------- #
# Delete the PDF from the directory:
os.remove(basc_pdf_filepath)

# Delete API Token from memory:
del token
# %%
# %%
#%% TEMP - BASC 3 VALIDITY INDEX TESTING

# for filename in df['basc3_pdf_filename']:
#     print(f'\nExtracting data from {filename}...')

#     with open(directory + filename, 'rb') as file:
#         pdf_reader = PdfReader(file)

#         # Find the string "Composite Score Summary" in the pdf and save the page number as css_page:
#         composite_score_page = None
#         scale_score_page = None
#         content_scale_score_page = None
#         validity_index_page = None

#         for i, page in enumerate(pdf_reader.pages):
#             text = page.extract_text()
#             if 'Composite Score Summary' in text:
#                 composite_score_page = i + 1
#             if 'Scale Score Summary' in text:
#                 scale_score_page = i + 1
#             if 'CONTENT SCALE SCORE TABLE:' in text:
#                 content_scale_score_page = i + 1
#             if 'VALIDITY INDEX SUMMARY' in text:
#                 validity_index_page = i + 1



# TODO: VALIDITY INDEX SUMMARY
#     # NOTE: NOT WORKING YET - VALIDITY TABLE IS SCREWING UP VALUES DEPENDING ON RATING TEXT
#     # Find the page number that contains the table titled: "Validity Index Summary":
#     data = tb.read_pdf(directory + filename, pages=validity_index_page, multiple_tables=True)
    
#     # Find the table number that contains the string "Response Pattern":
#     for i, table in enumerate(data):
#         if 'Response Pattern' in table.to_string():
#             table_number = i


#     if len(data[table_number]) == 1:
#         # Make data into a df:
#         vis_df = pd.DataFrame(data[table_number])
#         # Add a new row with blanks:
#         vis_df.loc[1] = ['','','']
#         vis_df.loc[2] = ['','','']

#         # Split the values in the first row into the second row at the string "\r":
#         vis_df.iloc[1,0] = vis_df.iloc[0,0].split('\r')[0]
#         vis_df.iloc[1,1] = vis_df.iloc[0,1].split('\r')[0]
#         vis_df.iloc[1,2] = vis_df.iloc[0,2].split('\r')[0]

#         # Split the values in the first row into the third row after the string "Raw Score:  ":
#         vis_df.iloc[2,0] = vis_df.iloc[0,0].split('Raw Score: ')[1]
#         vis_df.iloc[2,1] = vis_df.iloc[0,1].split('Raw Score: ')[1]
#         vis_df.iloc[2,2] = vis_df.iloc[0,2].split('Raw Score: ')[1]

#         # Remove the first row and reset index:
#         vis_df = vis_df.drop([0])
#         vis_df = vis_df.reset_index(drop=True)

#         # Return vis_df to data as a tb list:
#         data[table_number] = vis_df

#     if len(data[table_number]) == 2:

#     # If data[table_number] has 3 rows, then print ("boop")
#     elif len(data[table_number]) == 3:
#         # Remove the second row:
#         data[table_number] = data[table_number].drop([1])
#         # Reset the index:
#         data[table_number] = data[table_number].reset_index(drop=True)
#         # Replace the string "Raw Score: " with ""
#         data[table_number] = data[table_number].replace({'Raw Score:  ': ''}, regex=True)

#     # Replace the index for row 1 with "Rating:"
#     data[table_number].rename(index={0: 'Rating:'}, inplace=True)
#     # Replace the index for row 2 with "Raw Score:"
#     data[table_number].rename(index={1: 'Raw Score:'}, inplace=True)

#    # Replace the term "Acceptable" with 1:
#     data[table_number] = data[table_number].replace({'Acceptable': 1}, regex=True)
#     # Replace the term "Extreme Caution" with 3:
#     data[table_number] = data[table_number].replace({'Extreme Caution': 3}, regex=True)
#     # Replace the term "Caution" with 2:
#     data[table_number] = data[table_number].replace({'Caution': 2}, regex=True)


#     # Print the table:
#     print(data[table_number])

#     # Save the values into their corresponding columns in the df:
#     # F Index = basc_findex_rating, basc_findex_raw
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_findex_rating'] = data[table_number].iloc[0,0]
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_findex_raw'] = data[table_number].iloc[1,0]
#     # Response Pattern = basc_responsepattern_rating, basc_responsepattern_raw
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_responsepattern_rating'] = data[table_number].iloc[0,1]
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_responsepattern_raw'] = data[table_number].iloc[1,1]
#     # Consistency = basc_consistency_rating, basc_consistency_raw
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_consistency_rating'] = data[table_number].iloc[0,2]
#     df.loc[df['basc3_pdf_filename'] == filename, 'basc_consistency_raw'] = data[table_number].iloc[1,2]
