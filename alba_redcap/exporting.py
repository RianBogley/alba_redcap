# Export data from REDCap Token by Rian Bogley
# %%
# Import packages
import requests
import pandas as pd
import numpy as np
from io import StringIO

# %%
def export_data_redcap(token,data='raw',headers='raw'):
    """
    This function exports all data from a REDCap project using the project's API token.
    It can be used to export data in either raw or label format, and with either raw or label headers.
    """
    #!/usr/bin/env python
    data = {
        'token': token,
        'content': 'record',
        'action': 'export',
        'format': 'csv',
        'type': 'flat',
        'csvDelimiter': '',
        'rawOrLabel': data,
        'rawOrLabelHeaders': headers,
        'exportCheckboxLabel': 'false',
        'exportSurveyFields': 'true',
        'exportDataAccessGroups': 'false',
        'returnFormat': 'csv'
    }
    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    return df

def export_single_case_redcap(token, pidn, visit='visit_one_arm_1',data='raw',headers='label'):
    """
    This function exports a single case from REDCap using:
    - The REDCap project's API token.
    - The PIDN of the patient.
    - The specific visit number/event name. (Default is 'visit_one_arm_1').
    """
    #!/usr/bin/env python
    data = {
        'token': token,
        'content': 'record',
        'action': 'export',
        'format': 'csv',
        'type': 'flat',
        'csvDelimiter': '',
        'rawOrLabel': data,
        'rawOrLabelHeaders': headers,
        'exportCheckboxLabel': 'false',
        'exportSurveyFields': 'true',
        'exportDataAccessGroups': 'false',
        'returnFormat': 'csv',
        'records': pidn,
        'events': visit
    }
    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))
    print(r.text)
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    return df

def export_single_case_single_instrument_redcap(token, pidn, instrument, visit='visit_one_arm_1',data='raw',headers='raw'):
    """
    This function exports a single case from REDCap using:
    - The REDCap project's API token.
    - The PIDN of the patient.
    - The specific visit number/event name. (Default is 'visit_one_arm_1').
    """
    #!/usr/bin/env python
    data = {
        'token': token,
        'content': 'record',
        'action': 'export',
        'format': 'csv',
        'type': 'flat',
        'csvDelimiter': '',
        'records': pidn,
        'fields[0]':'id_number',
        'forms': instrument,
        'events': visit,
        'rawOrLabel': data,
        'rawOrLabelHeaders': headers,
        'exportCheckboxLabel': 'false',
        'exportSurveyFields': 'false',
        'exportDataAccessGroups': 'false',
        'returnFormat': 'csv'
    }

    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))
    print(r.text)
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    return df

def export_redcap_metadata(token):
    """
    This function exports the metadata/data dictionary from a REDCap project using the project's API token.
    The data dictionary can then be used to switch between variable names and labels, etc.
    """
    #!/usr/bin/env python
    data = {
        'token': token,
        'content': 'metadata',
        'format': 'csv',
        'returnFormat': 'csv'
    }
    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    return df

def export_redcap_file(token, pidn, visit='visit_one_arm_1', field='basc3_upload'):
    """
    This function exports a single file from a single case from REDCap using:
    - The REDCap project's API token.
    - The PIDN of the patient.
    - The specific visit number/event name. (Default is 'visit_one_arm_1').
    - The field name of the variable containing the file.

    """
    #!/usr/bin/env python
    data = {
        'token': token,
        'content': 'file',
        'action': 'export',
        'record': pidn,
        'field': field,
        'event': visit,
        'returnFormat': 'csv'
    }

    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))

    f = open('C:/Users/rbogley/Downloads/exported.pdf', 'wb')
    f.write(r.content)
    f.close()

    return f

