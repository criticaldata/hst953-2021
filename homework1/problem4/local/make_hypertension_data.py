# Generates the following data files from MIMIC:
# hypertension_patients.gz: patients with and without hypertension
# hypertension_charts.gz: chart events (heart rate, respiratory rate,
#   O2 saturation, blood pressure) for all patients
# Author: Bai Li (updated: Jan 3, 2019)

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
import os 
import random


# Ouput directory to generate the files
mimicdir = os.path.expanduser("./mimic_data")

random.seed(42)

# create a database connection
sqluser = 'mimicuser'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser, host='127.0.0.1', password='PASSWORD')
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)


# Select all patients ever admitted
all_patients_query = """
    select distinct subject_id, hadm_id from diagnoses_icd;
"""
all_patients = pd.read_sql_query(all_patients_query, con)


# Select patients with and without hypertension
patients_with_hypertension_query = """
    select subject_id, hadm_id from diagnoses_icd where icd9_code in ('4010', '4011', '4019');
"""
patients_with_hypertension = pd.read_sql_query(patients_with_hypertension_query, con)

all_patients['hypertension'] = 0
all_patients.loc[all_patients['hadm_id'].isin(patients_with_hypertension['hadm_id']), 'hypertension'] = 1


# Train test split
msk = np.random.rand(len(all_patients)) < 0.7
all_patients['train'] = np.where(msk, 1, 0) 

all_patients.to_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip', index = False)


# Pull chartevents
chartevents_query = """
    select subject_id, hadm_id, charttime, itemid, valuenum from chartevents where itemid in (220045, 220210, 220277, 220181);
"""

chartevents_table = pd.read_sql_query(chartevents_query, con)
chartevents_table.to_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip', index = False)

