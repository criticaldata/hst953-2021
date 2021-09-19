import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
import os
import random
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
import warnings
import yaml

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

mimicdir = "./mimic_data/"
Path(mimicdir).mkdir(parents = True, exist_ok = True)

random.seed(42)
np.random.seed(42)


# create a database connection
sqluser = 'mimicuser'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser, host='127.0.0.1', password='password')
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)


denquery = """
-- This query extracts useful demographic/administrative information for patient ICU stays
--DROP MATERIALIZED VIEW IF EXISTS icustay_detail CASCADE;
--CREATE MATERIALIZED VIEW icustay_detail as

--ie is the icustays table
--adm is the admissions table
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
, pat.gender
, adm.admittime, adm.dischtime, adm.diagnosis
, ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
, ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
, adm.ethnicity, adm.ADMISSION_TYPE, adm.language, adm.insurance
, adm.hospital_expire_flag
, CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
    ELSE 0 END AS first_hosp_stay
-- icu level factors
, ie.intime, ie.outtime
, ie.FIRST_CAREUNIT
, ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
    ELSE 0 END AS first_icu_stay

FROM icustays ie
INNER JOIN admissions adm
    ON ie.hadm_id = adm.hadm_id
INNER JOIN patients pat
    ON ie.subject_id = pat.subject_id
WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;

"""

den = pd.read_sql_query(denquery,con)

den['los_icu_hr'] = (den.outtime - den.intime).astype('timedelta64[h]')
den = den[(den.los_icu_hr >= 48)]
den = den[(den.age<300)]
den.drop('los_icu_hr', axis = 1, inplace = True)
den = den[(den.first_hosp_stay == 1) & (den.first_icu_stay == 1)]
den = den[~(den['first_careunit'].isin(['PICU', 'NICU']))]

den.ethnicity = den.ethnicity.str.lower()
den.ethnicity.loc[(den.ethnicity.str.contains('^white'))] = 'white'
den.ethnicity.loc[(den.ethnicity.str.contains('^black'))] = 'black'
den.ethnicity.loc[(den.ethnicity.str.contains('^hisp')) | (den.ethnicity.str.contains('^latin'))] = 'hispanic'
den.ethnicity.loc[(den.ethnicity.str.contains('^asia'))] = 'asian'
den.ethnicity.loc[~(den.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian'])))] = 'other'

den.drop(['diagnosis', 'hospstay_seq', 'los_icu','icustay_seq', 'los_hospital', 'outtime', 'first_careunit', 'first_hosp_stay', 'first_icu_stay'], axis = 1, inplace =True)

def map_lang(x):
    if x == 'ENGL':
        return 'English'
    if pd.isnull(x):
        return 'Missing'
    return 'Other'
den['language'] = den['language'].apply(map_lang)

with open('./icd9_codes.yml', 'r') as f:
    ccs = pd.DataFrame.from_dict(yaml.load(f)).T

icd = (pd.read_sql_query('select * from diagnoses_icd',con)
      .groupby(['subject_id','hadm_id'], as_index = False)
       .agg({'icd9_code': list})
      .merge(den[['subject_id', 'hadm_id']], on = ['subject_id', 'hadm_id'], how = 'inner')
      .explode('icd9_code'))

icd = (pd.merge(icd, (ccs[ccs['use_in_benchmark']]
                     .reset_index()
                     .explode('codes')), left_on = 'icd9_code', right_on = 'codes', how = 'inner')
      .rename(columns = {'index': 'name'}).drop(columns = ['use_in_benchmark', 'id', 'type', 'codes']))

targets = list(ccs[ccs['use_in_benchmark']].index)
target_df = icd.pivot_table(index = ['subject_id', 'hadm_id'], columns = 'name', values = 'icd9_code', aggfunc = lambda x: x)
target_df = (target_df.where(pd.isnull(target_df), 1)
             .fillna(0)
             .reset_index())

target_df['any acute'] = target_df[ccs[(ccs['use_in_benchmark']) & (ccs['type'] == 'acute')].index].any(axis = 1).astype(int)
target_df['any chronic'] = target_df[ccs[(ccs['use_in_benchmark']) & (ccs['type'] == 'chronic')].index].any(axis = 1).astype(int)

mapping = pd.read_csv('./mapping.csv').set_index('before')['after'].to_dict()
target_df = target_df.rename(columns = mapping)
den = pd.merge(den, target_df, how = 'left', on = ['subject_id', 'hadm_id'])

for col in mapping.values():
    den[col] = den[col].fillna(0)

notesquery = """
select row_id as note_id, subject_id, hadm_id, chartdate, charttime, category, text
from noteevents
where category in ('Discharge summary', 'Nursing', 'Nursing/other')
"""

notes = pd.read_sql_query(notesquery,con)
notes = notes[(notes.hadm_id.isin(den.hadm_id))]
notes['text'] = notes['text'].str.lower().apply(str.strip).fillna('')
notes['text'] = (notes['text'].str.replace(r'(-){2,}|_{2,}|={2,}', '')
                 .str.replace(r'[0-9]+\.', '')
                 .str.replace(r'\[(.*?)\]', '')
                 .str.replace(r'dr\.', 'doctor')
                 .str.replace(r'm\.d\.', 'md')
                 .str.replace(r'admission date:', '')
                 .str.replace(r'discharge date:', '')
                 .str.replace(r'\n', ' ')
                 .str.replace(r'\r', ' ')
                )

den = den[den['subject_id'].isin(notes['subject_id'])] #drop people with no notes

msk = np.random.rand(len(den)) < 0.7
den['train'] = np.where(msk, 1, 0)
den.to_hdf(os.path.join(mimicdir, 'cohort.h5'), 'cohort', index = False)
notes.to_hdf(os.path.join(mimicdir, 'notes.h5'), 'notes', index = False)
