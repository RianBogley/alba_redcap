# Import data back to REDCap via API Token by Rian Bogley
# %%
# Import packages
import requests
import pandas as pd
import numpy as np
from io import StringIO

# %%
def import_data_redcap(token,df,data='raw',headers='raw',overwriteBehavior='normal'):
    """
    This function exports all data from a REDCap project using the project's API token.
    It can be used to export data in either raw or label format, and with either raw or label headers.
    """
    #!/usr/bin/env python
    # Convert dataframe to csv format:
    converted_df = df.to_csv(index=False)

    data = {
        'token': token,
        'content': 'record',
        'action': 'import',
        'format': 'csv',
        'type': 'flat',
        'overwriteBehavior': overwriteBehavior,
        'forceAutoNumber': 'false',
        'data': converted_df,
        'dateFormat': 'MDY',
        'returnContent': 'count',
        'returnFormat': 'csv',
        'rawOrLabel': data,
        'rawOrLabelHeaders': headers,
    }
    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    print('HTTP Status: ' + str(r.status_code))

    # Return any errors:
    if r.status_code != 200:
        print(r.text)
    else:
        print('REDCap Import Successful!')