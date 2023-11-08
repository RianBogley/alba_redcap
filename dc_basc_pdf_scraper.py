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
import numpy as np
import openpyxl
import pandas as pd
import pingouin as pg
import plotly as py
import ptitprince as pt
import requests
import scipy.stats as stats
import seaborn as sns
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tabula as tb
from joblib import Parallel, delayed
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
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
redcap_tools_path = code_dir + 'alba-redcap-tools'

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
pidn = int(pidn)
visit = input("Enter REDCap Visit Number to export (e.g. 'visit_one_arm_1'):")
# ---------------------------------------------------------------------------- #
# Set instrument of interest:
instrument = 'basc'
# Export the specified instrument for that case:
basc_df = export_single_case_single_instrument_redcap(token, pidn, instrument, visit)
# Replace all nans with blanks in the df:
basc_df = basc_df.replace(np.nan, '', regex=True)

# ---------------------------------------------------------------------------- #
# Check which BASC Version is specified in REDCap:
if basc_df['basc_version'].values[0] == 3:
    field = 'basc3_upload'
    print('BASC Version 3 Specified.')
elif basc_df['basc_version'].values[0] == 2:
    field = 'basc2_upload'
    print('BASC Version 2 Specified.')
elif basc_df['basc_version'].values[0] == '':
    # Check for uploaded BASC-3 or BASC-2 PDFs:
    if basc_df['basc3_upload'].values[0] != '' and basc_df['basc2_upload'].values[0] == '':
        print(f'BASC-3 PDF found. Extracting data from {basc_df["basc3_upload"].values[0]}...')
        field = 'basc3_upload'
        basc_df['basc_version'].values[0] = 3
    elif basc_df['basc3_upload'].values[0] == '' and basc_df['basc2_upload'].values[0] != '':
        print(f'BASC-2 PDF found. Extracting data from {basc_df["basc2_upload"].values[0]}...')
        field = 'basc2_upload'
        basc_df['basc_version'].values[0] = 2
    elif basc_df['basc3_upload'].values[0] != '' and basc_df['basc2_upload'].values[0] != '':
        print('ERROR: Both BASC-3 and BASC-2 PDFs found.')
        print('Please ensure the REDCap BASC page only has one uploaded file per visit or specify which version to choose in "basc_version" and try again.')
        sys.exit()
    elif basc_df['basc3_upload'].values[0] == '' and basc_df['basc2_upload'].values[0] == '':
        print('ERROR: No BASC-3 or BASC-2 PDFs found.')
        print('Please upload the BASC-3 or BASC-2 PDF to REDCap and try again.')
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

# ---------------------------------------------------------------------------- #
#                               DEFINE FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

# TODO: SCRIPT NOT SAVING SOME VALUES!
# TODO: CHECK IF THE PDF IF SELF-REPORT

def find_page_index(pdf_reader, keywords):
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if any(keyword in text for keyword in keywords):
            return i

def find_table_index(data, keywords):
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
    # NOTE: TODO: ONLY CHANGES FIRST ROW OF THE DF, DOES NOT WORK FOR MULTIPLE PIDNS
    for variable in variables_dict:
        # Save the raw score to the first row of the dataframe:
        df[variables_dict[variable] + '_raw'] = table.loc[table['Domain'] == variable, 'Raw Score'].values[0]
        # Save the t score to the first row of the dataframe:
        df[variables_dict[variable] + '_tscore'] = table.loc[table['Domain'] == variable, 'T Score'].values[0]
        # Save the percentile to the first row of the dataframe:
        df[variables_dict[variable] + '_per'] = table.loc[table['Domain'] == variable, 'Percentile'].values[0]
    return df

def extract_page_text(pdf_reader, page_number):
    # Import all the text on the specified page of the pdf:
    full_text = pdf_reader.pages[page_number].extract_text()
    # Replace all new line characters with spaces:
    full_text = full_text.replace('\n', ' ')
    # Return the text on the page
    return full_text

def find_string_in_text(full_text, keyword_start, keyword_end):
    # Check if both keywords exist in the text:
    if keyword_start not in full_text or keyword_end not in full_text:
        print(f'ERROR: {keyword_start} not found.')
        print(f'Please check the pdf and try again.')
        pass
    else:
        # Find the start and end indices of the text between the keywords:
        start_index = full_text.find(keyword_start) + len(keyword_start)
        end_index = full_text.find(keyword_end)
        # Extract the text between the keywords:
        text = full_text[start_index:end_index]
        # Remove any trailing spaces:
        text = text.strip()
        # Return the text:
        return text

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                             READ IN THE PDF DATA                             #
# ---------------------------------------------------------------------------- #
with open(basc_pdf_filepath, 'rb') as file:
    pdf_reader = PdfReader(file)
    
    # Find the page numbers of interest:
    date_page = find_page_index(pdf_reader, ['Test Date:'])
    comments_concerns_page = find_page_index(pdf_reader, ['COMMENTS AND CONCERNS', 'Rater Concerns'])
    composite_score_page = find_page_index(pdf_reader, ['Composite Score Summary'])
    scale_score_page = find_page_index(pdf_reader, ['Scale Score Summary'])
    content_scale_score_page = find_page_index(pdf_reader, ['CONTENT SCALE SCORE TABLE:','The information provided below is'])
    validity_index_page = find_page_index(pdf_reader, ['VALIDITY INDEX SUMMARY', 'Validity Index Summary'])

data = tb.read_pdf(basc_pdf_filepath, pages='all', multiple_tables=True)

# FIND BASC DATE STRING IN TEXT:
if date_page is not None:
    # Extract the page text:
    date_text = extract_page_text(pdf_reader, date_page)

    # Set the keywords:
    keyword_date_start = 'Test Date:'
    keyword_date_end = 'Rater'

    # Find the date string in the text:
    basc_date = find_string_in_text(date_text, keyword_date_start, keyword_date_end)
    # Remove any text in the basc_date string after the first space:
    basc_date = basc_date.split(' ')[0]
    # Set basc_date to YYYY-MM-DD format:
    basc_date = datetime.datetime.strptime(basc_date, '%m/%d/%Y').strftime('%Y-%m-%d')
    print(f'BASC Date Found: {basc_date}')

    # Apply date to appropriate column in dataframe:
    if basc_version == 3:
        basc_df['basc3_date'] = basc_date
    elif basc_version == 2:
        basc_df['basc2_date'] = basc_date

# COMMENTS AND CONCERNS TEXT:
if comments_concerns_page is not None:
    # Extract the page text:
    comments_concerns_text = extract_page_text(pdf_reader, comments_concerns_page)

    # Set the keywords:
    keyword_rater_concerns_start = 'Rater Concerns'
    keyword_rater_concerns_end = 'Rater General Comments'
    keyword_rater_general_strengths = 'What are the behavioral and/or emotional strengths of this child?'
    keyword_rater_general_concerns = 'Please list any specific behavioral and/or emotional concerns you have about this child.'
    keyword_comments_concerns_end = 'BASC™-'

    # Rater Concerns:
    basc_rater_concerns = find_string_in_text(comments_concerns_text, keyword_rater_concerns_start, keyword_rater_concerns_end)
    print(f'Rater Concerns: {basc_rater_concerns}')
    basc_df = basc_df.assign(basc_rater_concerns = basc_rater_concerns)

    # Rater General Strengths:
    basc_general_strengths = find_string_in_text(comments_concerns_text, keyword_rater_general_strengths, keyword_rater_general_concerns)
    print(f'Rater General Strengths: {basc_general_strengths}')
    basc_df = basc_df.assign(basc_general_strengths = basc_general_strengths)

    # Rater General Concerns:
    basc_general_concerns = find_string_in_text(comments_concerns_text, keyword_rater_general_concerns, keyword_comments_concerns_end)
    print(f'Rater General Concerns: {basc_general_concerns}')
    basc_df = basc_df.assign(basc_general_concerns = basc_general_concerns)

# COMPOSITE SCORE SUMMARY TABLE:
if composite_score_page is not None:
    composite_score_variables = {'Externalizing Problems': 'basc_externalizing',
                                    'Internalizing Problems': 'basc_internalizing',
                                    'Behavioral Symptoms Index': 'basc_behavior',
                                    'Adaptive Skills': 'basc_adaptiveskill'}
    # Find the table:
    composite_score_table = data[find_table_index(data, composite_score_variables.keys())]
    # Clean the table:
    composite_score_table = clean_table(composite_score_table)
    print(composite_score_table)

    # Save the table values to the dataframe:
    basc_df = save_table_values(basc_df, composite_score_table, composite_score_variables)

# SCALE SCORE SUMMARY TABLE:
if scale_score_page is not None:
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
    scale_score_table = data[find_table_index(data, scale_score_variables.keys())]
    # Clean the table:
    scale_score_table = clean_table(scale_score_table)
    print(scale_score_table)
    # Save the table values to the dataframe:
    basc_df = save_table_values(basc_df, scale_score_table, scale_score_variables)

# CONTENT SCALE SCORE TABLE:
if content_scale_score_page is not None:
    content_scale_score_variables = {'Anger Control': 'basc_angercontrol',
                                        'Bullying': 'basc_bullying',
                                        'Developmental Social Disorders': 'basc_socialdisord',
                                        'Emotional Self-Control': 'basc_emotioncontrol',
                                        'Executive Functioning': 'basc_execfunc',
                                        'Negative Emotionality': 'basc_negemotion',
                                        'Resiliency': 'basc_resiliency'}
    # Find the table:
    content_scale_score_table = data[find_table_index(data,[key for key in content_scale_score_variables.keys() if key != 'Developmental Social Disorders'])]
    # Clean the table:
    content_scale_score_table = clean_table(content_scale_score_table)
    print(content_scale_score_table)
    # Save the table values to the dataframe:
    basc_df = save_table_values(basc_df, content_scale_score_table, content_scale_score_variables)

# MARK BASC INSTRUMENT AS COMPLETE
basc_df['basc_complete'] = 2

# ---------------------------------------------------------------------------- #
#                     EXPORT DATA TO CSV FOR REDCAP IMPORT                     #
# ---------------------------------------------------------------------------- #
# # Find parent directory of basc_filepath:
# output_dir = os.path.dirname(basc_pdf_filepath)
# # Export the basc_df to csv:
# basc_df.to_csv(f'{output_dir}/{pidn}_{visit}_basc-{basc_version}.csv', index=False)


# ---------------------------------------------------------------------------- #
#                        IMPORT THE DATA BACK TO REDCAP                        #
# ---------------------------------------------------------------------------- #
# WARNING: THIS WILL OVERWRITE EXISTING DATA IN REDCAP

# Import the basc_df back to REDCap via API using import_data_redcap:
import_data_redcap(token, basc_df, data='raw', headers='raw', overwriteBehavior='normal')

# ---------------------------------------------------------------------------- #
#                                 FINISHING UP                                 #
# ---------------------------------------------------------------------------- #
# Delete the PDF from the directory:
os.remove(basc_pdf_filepath)

# Delete Token, PIDN, and Visit Number from memory:
del token
del pidn
del visit

# %%

# TODO: VALIDITY INDEX SUMMARY TABLE IS TOO WACK:

# def clean_validity_index(table):
#     # If there are multiple rows, merge the values in the first and last rows
#     # as long as neither value is NaN:
#     if table.shape[0] > 1:
#         if pd.notna(table.iloc[-1,0]):
#             table.iloc[0,0] = str(table.iloc[0,0]) + "\r" + str(table.iloc[-1,0])
#         if pd.notna(table.iloc[-1,1]):
#             table.iloc[0,1] = str(table.iloc[0,1]) + "\r" + str(table.iloc[-1,1])
#         if pd.notna(table.iloc[-1,2]):
#             table.iloc[0,2] = str(table.iloc[0,2]) + "\r" + str(table.iloc[-1,2])

#         # Drop all rows except the first:
#         table = table.drop(table.index[1:])

#     # Make a new row and add the column names to the new row as values:
#     table.loc[1] = ['','','']
#     # Add the column names to the new row as values:
#     table.iloc[1,0] = table.columns[0]
#     table.iloc[1,1] = table.columns[1]
#     table.iloc[1,2] = table.columns[2]
    
#     # Create another row for the table and split the values in the first row into the new row at the string "\r":
#     table.loc[2] = ['','','']
#     table.iloc[2,0] = table.iloc[0,0].split('\r')[0]
#     table.iloc[2,1] = table.iloc[0,1].split('\r')[0]
#     table.iloc[2,2] = table.iloc[0,2].split('\r')[0]

#     # Create another row and take the values from the first row after "Raw Score: "
#     table.loc[3] = ['','','']
#     table.iloc[3,0] = table.iloc[0,0].split('Raw Score: ')[1]
#     table.iloc[3,1] = table.iloc[0,1].split('Raw Score: ')[1]
#     table.iloc[3,2] = table.iloc[0,2].split('Raw Score: ')[1]

#     # Remove the first row:
#     table = table.drop([0])

#     # Flip the table:
#     table = table.transpose()

#     # Remove trailing or tailing spaces in the whole table:
#     table = table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#     # Reset the index:
#     table.reset_index(drop=True, inplace=True)

#     # Set the first column name to "Domain"
#     table.rename(columns={table.columns[0]: 'Domain'}, inplace=True)
#     # Set the second column name to "Rating"
#     table.rename(columns={table.columns[1]: 'Rating'}, inplace=True)
#     # Set the third column name to "Raw Score"
#     table.rename(columns={table.columns[2]: 'Raw Score'}, inplace=True)

#     return table

# def save_validity_index(df, table, variables_dict):
#     #NOTE:TODO: THIS MAY OVERWRITE ALL THE VALUES IN THE DATAFRAME FOR THE GIVEN COLUMNS, ONLY WORKS FOR ONE PIDN FOR NOW
#     # Change the rating values to numbers:
#     validity_index_rating_scores = {'Acceptable': 1,
#                                     'Caution': 2,
#                                     'Extreme Caution': 3}
#     # Change the rating scores to numbers using the validity_index_rating_scores dictionary:
#     table['Rating'] = table['Rating'].map(validity_index_rating_scores)

#     for variable in variables_dict:
#         # Save the rating to the dataframe to the column that matches the variable name + "_rating":
#         df[variables_dict[variable] + '_rating'] = table.loc[table['Domain'] == variable, 'Rating'].values[0]
#         # Save the raw score to the dataframe to the column that matches the variable name + "_raw":
#         df[variables_dict[variable] + '_raw'] = table.loc[table['Domain'] == variable, 'Raw Score'].values[0]
#     return df



# # VALIDITY INDEX SUMMARY TABLE:
# if validity_index_page is not None:    
#     if basc_version == 3:
#         validity_index_variables = {'F Index': 'basc_findex',
#                                     'Response Pattern': 'basc_responsepattern',
#                                     'Consistency': 'basc_consistency'}
#     elif basc_version == 2:
#         validity_index_variables = {'F': 'basc_findex',
#                                     'Response Pattern': 'basc_responsepattern',
#                                     'Consistency': 'basc_consistency'}
#     # Find the table:
#     validity_index_table = data[find_table_index(data, validity_index_variables.keys())]
#     # Clean the table:
#     validity_index_table = clean_validity_index(validity_index_table)
#     print(validity_index_table) 
#     # Save the table values to the dataframe:
#     basc_df = save_validity_index(basc_df, validity_index_table, validity_index_variables)

# %%
