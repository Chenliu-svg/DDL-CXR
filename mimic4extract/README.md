MIMIC-IV Data Extraction
=========================

Here we modified codes from the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/) and [MedFuse](https://github.com/nyuad-cai/MedFuse) for extract and process EHR data for our task.

## Download the MIMIC-IV

We do not provide the MIMIC-IV data itself. You must acquire the data yourself from https://physionet.org/content/mimiciv/2.0/. Specially, download and extract the CSV files.


## Pre-process the EHR data

1. The following command takes MIMIC-IV CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`.  
       ```bash
       cd ./mimic4extract

       mimic_iv_csv_dir="/path/to/mimic_iv_csv" # e.g. /root/autodl-tmp/0807_version/physionet.org/files/mimiciv/2.0/
       mimic_iv_subjects_dir="path/to/save/mimic-iv/subjects/ehr" # e.g. /root/autodl-tmp/dynamic-prediction/data/mimic_iv_subjects_dir

       python -m mimic3benchmark.scripts.extract_subjects_iv $mimic_iv_csv_dir $mimic_iv_subjects_dir
       ```
Meanwhile, after this process ICU stays of all the subjects has been save in `$mimic_iv_subjects_dir/all_stays.csv`. And phenotype labels of all the subjects has been save in `$mimic_iv_subjects_dir/phenotype_labels.csv`.

2. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

       ```bash
       python -m mimic3benchmark.scripts.validate_events $mimic_iv_subjects_dir
       ```
3. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{ICU_STAY_ID}_timeseries.csv```  while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, diagnoses) are stores in ```{SUBJECT_ID}/episode{ICU_STAY_ID}.csv```. 
       ```bash
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects $mimic_iv_subjects_dir
       ```
4. After the steps above, each subject in `$mimic_iv_subjects_dir` has the directory tree as follows:
       
```bash
{SUBJECT_ID}
|---diagnoses.csv  #(Fields: subject_id,hadm_id,seq_num,icd_code,icd_version,long_title,stay_id,HCUP_CCS_2015,USE_IN_BENCHMARK)
|---events.csv  #(Fields:subject_id,hadm_id,stay_id,charttime,itemid,value,valuenum)
|---stays.csv #(Fields:subject_id,hadm_id,stay_id,last_careunit,intime,outtime,los,admittime,dischtime,deathtime,gender,anchor_age,dod,age,mortality_inunit,mortality,mortality_inhospital)
|---episode{ICU_STAY_ID_1}.csv       #(Fields:Icustay,Gender,Age,Height,Weight,Length of Stay,Mortality,Diagnosis I169,Diagnosis I509,Diagnosis I2510,Diagnosis I4891...)
|---episode{ICU_STAY_ID_1}_timeseries.csv #(Fields:Hours,Capillary refill rate,Diastolic blood pressure,Fraction inspired oxygen,Glascow coma scale eye opening,Glascow coma scale motor response,Glascow coma scale total,Glascow coma scale verbal response,Glucose,Heart Rate,Height,Mean blood pressure,Oxygen saturation,Respiratory rate,Systolic blood pressure,Temperature,Weight,ph)
|---episode{ICU_STAY_ID_2}_timeseries.csv
|---episode{ICU_STAY_ID_2}.csv  # A subject might have multuples icu stays.
```
