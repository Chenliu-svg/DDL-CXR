"""
pre_process.py --mimic_cxr_jpg_dir {mimic_cxr_jpg_dir} --mimic_iv_csv_dir {mimic_iv_csv_dir} --mimic_iv_subjects_dir {mimic_iv_subjects_dir} --output_csv_dir {output_csv_dir}

args:
    --mimic_cxr_jpg_dir: The path to the mimic-cxr-jpg-dataset, where the `mimic-cxr-2.0.0-metadata.csv`,`mimic-cxr-2.0.0-chexpert.csv` is.
    --mimic_iv_csv_dir: The path to the original mimic-iv csv files, where the `hosp/patients.csv` is.
    --mimic_iv_subjects_dir: The path to the processed mimic-iv subjects files, where the `all_stay.csv`,`phenotype_labels.csv`, `{subject_id}/episode{stay_id}_timeseries.csv` is.
    --output_csv_dir: The path to the output processed csv files(data and lables).

Output:
    1. {train/validate/test}_autoencoder_augmented_label.csv: data and label for train/validate/test split of Autoencoder
        subject_id,study_id,dicom_id,{Chexpert labels for the CXR}
    2. {train/validate/test}_dm_labels.csv: data and label for train/validate/test split of DM
        subject_id,x0_study_id,x1_study_id,x0_dicom_id,x1_dicom_id,stay_id,ehr_len,left_time,right_time,mortality_inhospital,Gender,Age,{Chexpert labels of x0},{Chexpert labels of x1}
    3. {train/validate/test}_pred.csv: data for train/validate/test split of prediction
        subject_id,stay_id,study_id,dicom_id,mortality_inhospital,Gender,Age,cha(time difference(in hours) between the last cxr and the prediction time),{Phenotype labels of the icu stay}
"""
# import necessary packages
import pandas as pd
import math
from datetime import timedelta
import os
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# some frequently used variable
zero_timedelta = timedelta(0)
one_day=timedelta(days=1)

disease_cols=['Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity',  'Pleural Effusion',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding']
cols=['subject_id','study_id']+disease_cols

def check_ehr_length_distribution(neighbor_cxr_meta_path,delta,all_stay_path,mimic_iv_subjects_dir):
    """
    Print ehr length distribution with neighbor cxr taken at least delta hours apart
    
    Args:
        neighbor_cxr_meta_path: meta csv file path, with fields: subject_id, stay_id, x0_time, x0_dicom_id, x1_time, x1_dicom_id, cha(time difference(in hours) between x0 and x1)
        delta: minimum time difference(in hours) between x0 and x1
        mimic_iv_subjects_dir: mimic-iv subject directory
        all_stay_path: Icu stay information of all subjects csv file path
        
    """
    dm_df=pd.read_csv(neighbor_cxr_meta_path)
    all_stay=pd.read_csv(all_stay_path)
    
    dm_df_selected=dm_df[dm_df.cha>delta]
    dm_df_selected=pd.merge(dm_df_selected,all_stay,on=['subject_id','stay_id'], how='left')[['subject_id','stay_id','x0_time', 'x0_dicom_id', 'x1_time',
        'x1_dicom_id','intime']]
    

    timestep=1
    ehr_len=[]

    for index, row in dm_df_selected.iterrows():
        patient=row.subject_id
        stay_id=row.stay_id
        l_interval=row.x0_time
        r_interval=row.x1_time
        intime =row.intime
        ts_df=pd.read_csv(os.path.join(mimic_iv_subjects_dir, str(patient), f'episode{stay_id}_timeseries.csv'))
        ts_df_interval=ts_df[(ts_df.Hours>= (l_interval-intime)/one_day*24 )&(ts_df.Hours <= (r_interval -intime)/one_day*24)].reset_index(drop=True)
        if len(ts_df_interval)==0:
            # print(stay_id)
            # empty+=1
            ehr_len.append(0)
            continue
        start=ts_df_interval.loc[0,'Hours']
        interval=ts_df_interval['Hours'].apply(lambda x:((x-start)//timestep))
        interval=interval.drop_duplicates()
        ehr_len.append(len(interval))
        
    len_df=pd.DataFrame(data={"ehr_len":ehr_len})
    print(f'ehr length distribution with neighbor cxr taken at least {delta} hours : \n',len_df.describe())



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # CXR related files
    parser.add_argument('--mimic_cxr_jpg_dir', required=True, help='The path to the mimic-cxr-jpg-dataset')
    parser.add_argument('--mimic_iv_subjects_dir', required=True, help='The path to the mimic-iv subjects ')
    parser.add_argument('--mimic_iv_csv_dir', required=True, help='The path to the origin mimic-iv CSVs ')
    parser.add_argument('--output_csv_dir', required=True, help='The path for the output CSV from the pre-processing ')
    parser.add_argument('--random_seed', default=1, help='random seed ')
    
    
    args = parser.parse_args()
    for arg in vars(args):
        print('{}\t{}'.format(arg, getattr(args, arg)))
    
    if not os.path.exists(args.output_csv_dir):
        os.makedirs(args.output_csv_dir)
    
    # read all icu stay records
    all_stay=pd.read_csv(os.path.join(args.mimic_iv_subjects_dir,'all_stays.csv'))

    # convert time
    all_stay.intime=pd.to_datetime(all_stay.intime)
    all_stay.outtime=pd.to_datetime(all_stay.outtime)
    all_stay.dod=pd.to_datetime(all_stay.dod)
    all_stay.deathtime=pd.to_datetime(all_stay.deathtime)
    
    # remove patients who stayed for less than 48 hours
    all_stay=all_stay[all_stay.los*24>48]
    
    # check the uniqueness of the stay_id
    assert all_stay.stay_id.nunique()==len(all_stay)
   
    
    ## handling cxr: get the latest AP  for each study case
    cxr=pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-metadata.csv'))
    
    chexpert=pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir,'mimic-cxr-2.0.0-chexpert.csv'))[cols]
          
    # convert time to pd.datetime
    cxr['StudyTime'] = cxr['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr['StudyDateTime'] = pd.to_datetime(cxr['StudyDate'].astype(str) + ' ' + cxr['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    # get the AP
    # print(len(cxr[(cxr['ViewPosition']=='AP')]))
    cxr_AP=cxr[(cxr['ViewPosition']=='AP')].dropna(subset=['ViewPosition'])
    
    
    # get the latest AP for each study case
    cxr_AP_sorted=cxr_AP.sort_values(['subject_id','study_id','ViewPosition','StudyDateTime'],ascending=True)
    cxr_latest_AP=cxr_AP_sorted.groupby(['subject_id','study_id','ViewPosition']).nth(-1).reset_index()
    

    # link cxr with icu stay
    AP_merged_icustays = cxr_latest_AP.merge(all_stay, how='inner', on='subject_id')
    # get cxr taken in the first 48 hours in icu stay or in the last 24 hours in ED
    AP_bf_selected = AP_merged_icustays.loc[
    (((AP_merged_icustays.StudyDateTime-AP_merged_icustays.intime)/one_day * 24) <=48)
    
    &(((AP_merged_icustays.StudyDateTime-AP_merged_icustays.intime)/one_day*24)>=-24)
    ]
    # print(AP_bf_selected.stay_id.nunique())
    # print(all_stay.stay_id.nunique())
    
    # choose the latest one for each icu stay
    AP_bf_selected_sorted=AP_bf_selected.sort_values(['subject_id','stay_id','StudyDateTime'],ascending=True)
    AP_bf_selected_latest=AP_bf_selected_sorted.groupby(['subject_id','stay_id']).nth(-1).reset_index()
    AP_bf_selected_latest['cha']=AP_bf_selected_latest[['intime','StudyDateTime']].apply(lambda x:(2-(x['StudyDateTime']-x['intime'])/one_day)*24,axis=1)
    AP_bf_selected_small=AP_bf_selected_latest[['subject_id','stay_id','StudyDateTime','dicom_id','study_id']]
        
    
    
    ############################## Create the datasets for each steps ##############################
    
    random_seed=args.random_seed
    small_latest=AP_bf_selected_latest[['subject_id','stay_id','study_id','dicom_id','cha','mortality_inhospital']]
    
    eps=0.03
    
    mort_rate=len(small_latest[small_latest.mortality_inhospital==0])/len(small_latest[small_latest.mortality_inhospital==1])

    ### split by subject id
    subject_id=list(small_latest.subject_id.unique())
    train_val,test=train_test_split(subject_id, test_size=0.2,random_state=random_seed)
    train,val=train_test_split(train_val, test_size=1/7, random_state=random_seed)

    train_df=pd.DataFrame(data={'subject_id':train})
    val_df=pd.DataFrame(data={'subject_id':val})
    test_df=pd.DataFrame(data={'subject_id':test})
    
    ### Build the dataset for prediction
    train_df=pd.merge(train_df,small_latest,on='subject_id',how='left')
    val_df=pd.merge(val_df,small_latest,on='subject_id',how='left') 
    test_df=pd.merge(test_df,small_latest,on='subject_id',how='left')

    print('********************************')
    print('icu stay number')
    print(f'icu stay for train: {train_df.stay_id.nunique()}')
    print(f'icu stay for val: {val_df.stay_id.nunique()}')
    print(f'icu stay for test: {test_df.stay_id.nunique()}')

    print('********************************')
    print('mort distribute')
    print(f'mort total: {mort_rate}')
    print(sum(train_df['mortality_inhospital']==0)/sum(train_df['mortality_inhospital']==1))
    print(sum(val_df['mortality_inhospital']==0)/sum(val_df['mortality_inhospital']==1))
    print(sum(test_df['mortality_inhospital']==0)/sum(test_df['mortality_inhospital']==1))
    
    splitted_df={'train':train_df,'validate':val_df,'test':test_df}
    
    # add static info from patients.csv
    patients=pd.read_csv(os.path.join(args.mimic_iv_csv_dir,'hosp/patients.csv'))
    # add phenotype labels 
    pheno_df=pd.read_csv(os.path.join(args.mimic_iv_subjects_dir,'phenotype_labels.csv'))
    pheno_cols=pheno_df.columns.tolist()
    cols_tosave=['subject_id', 'stay_id','study_id', 'dicom_id','mortality_inhospital',"Gender","Age",'cha']+pheno_cols
    for s in ['train','validate','test']:
        pred_df_add_satic=pd.merge(splitted_df[s],patients,on='subject_id',how='left')
        pred_df_add_satic['Age']=pred_df_add_satic.anchor_age
        pred_df_add_satic.loc[pred_df_add_satic.Age<0,'Age']=90
        pred_df_add_satic['Gender']=pred_df_add_satic.gender
        pred_df_add_satic.loc[pred_df_add_satic.Gender=='F','Gender']=0
        pred_df_add_satic.loc[pred_df_add_satic.Gender=='M','Gender']=1
        pred_df_add_satic_add_pheno=pd.merge(pred_df_add_satic,pheno_df,on='stay_id',how='left')
        pred_df_add_satic_add_pheno=pred_df_add_satic_add_pheno[cols_tosave].reset_index()
        
        pred_df_add_satic_add_pheno.to_csv(os.path.join(args.output_csv_dir,f'{s}_pred.csv'))

    
    ### Build the dataset for DM
    from itertools import combinations
    result={'subject_id':[],'stay_id':[],'x0_time':[],'x0_dicom_id':[],'x1_time':[],'x1_dicom_id':[],}
    
    def create_combinations(group):
        # print(group)
        if len(group)>1:
            for pair in combinations(group.index,2):
                # print(pair)
                
                x0 =group.loc[pair[0]]
                x1 =group.loc[pair[1]]
                result['subject_id'].append(x0.subject_id)
                result['stay_id'].append(x0.stay_id)
                result['x0_time'].append(x0.StudyDateTime)
                result['x1_time'].append(x1.StudyDateTime)
                result['x0_dicom_id'].append(x0.dicom_id)
                result['x1_dicom_id'].append(x1.dicom_id)
    
    
    AP_bf_selected_small_dm=AP_bf_selected[['subject_id','stay_id','StudyDateTime','dicom_id']]
    AP_bf_selected_sorted_dm=AP_bf_selected_small_dm.sort_values(['subject_id','stay_id','StudyDateTime'],ascending=True)

    AP_bf_selected_sorted_dm.groupby(['subject_id', 'stay_id']).apply(create_combinations)

    dm_df=pd.DataFrame(data=result)
    dm_df['cha']=dm_df[["x0_time","x1_time"]].apply(lambda x:(x["x1_time"]-x["x0_time"])/one_day*24, axis=1)
    dm_df.to_csv(f'{args.output_csv_dir}/neighbor_cxr_pairs_meta.csv')
    
    dm_df_selected=dm_df[dm_df.cha>12]
    
    # get ehr length field
    
    dm_df_selected=pd.merge(dm_df_selected,all_stay,on=['subject_id','stay_id'], how='left')[['subject_id','stay_id','x0_time', 'x0_dicom_id', 'x1_time',
        'x1_dicom_id','intime']]


    timestep=1
    dm_df_selected['ehr_len']=0

    for index, row in dm_df_selected.iterrows():
        patient=row.subject_id
        stay_id=row.stay_id
        l_interval=row.x0_time
        r_interval=row.x1_time
        intime =row.intime
        ts_df=pd.read_csv(os.path.join(args.mimic_iv_subjects_dir, str(patient), f'episode{stay_id}_timeseries.csv'))
        ts_df_interval=ts_df[(ts_df.Hours>= (l_interval-intime)/one_day*24 )&(ts_df.Hours <= (r_interval -intime)/one_day*24)].reset_index(drop=True)
        if len(ts_df_interval)==0:
            ehr_len=0
            
        else:
            start=ts_df_interval.loc[0,'Hours']
            interval=ts_df_interval['Hours'].apply(lambda x:((x-start)//timestep))
            interval=interval.drop_duplicates()
            ehr_len=len(interval)
            
        dm_df_selected.loc[index,'ehr_len']=ehr_len
        
    # delete cxr pairs with ehr_len<6
    dm_df_selected_ehr=dm_df_selected[dm_df_selected.ehr_len>=6]
    
    disease_cols_dm=[i+'_l0' for i in disease_cols]+[i+'_l1' for i in disease_cols]
    print("********************************")
    for s in ['train','validate','test']:
        dm_split=pd.merge(splitted_df[s],dm_df_selected_ehr,
                        on=['subject_id','stay_id'],how='left')
        
        # # for prediction one cxr is okay, but for dm training it is not okay
        # # so there will be nan
        dm_split.dropna(inplace=True)
        dm_split=pd.merge(dm_split, cxr_AP,left_on=['subject_id','x0_dicom_id'],
                            right_on=['subject_id','dicom_id'], how='left')
        
        dm_split=pd.merge(dm_split, cxr_AP,left_on=['subject_id','x1_dicom_id'],
                            right_on=['subject_id','dicom_id'], how='left')
        
        dm_split.rename(columns={'study_id_y':'x0_study_id','study_id':'x1_study_id'},inplace=True)
        
        dm_split['left_time']=dm_split[['x0_time','intime']].apply(lambda x:(x['x0_time']-x['intime'])/one_day*24, axis=1)
        dm_split['right_time']=dm_split[['x1_time','intime']].apply(lambda x:(x['x1_time']-x['intime'])/one_day*24, axis=1)

        dm_split_small=dm_split[['subject_id',
                                'x0_study_id', 'x1_study_id',
                                'x0_dicom_id','x1_dicom_id',
                                'stay_id','ehr_len',
                                'left_time','right_time',
                                'mortality_inhospital'

                                ]].reset_index()
                                                
            

        dm_split_add_satic=pd.merge(dm_split_small,patients,on='subject_id',how='left')
        dm_split_add_satic.columns
        dm_split_add_satic['Age']=dm_split_add_satic.anchor_age
        dm_split_add_satic.loc[dm_split_add_satic.Age<0,'Age']=90
        dm_split_add_satic['Gender']=dm_split_add_satic.gender
        dm_split_add_satic.loc[dm_split_add_satic.Gender=='F','Gender']=0
        dm_split_add_satic.loc[dm_split_add_satic.Gender=='M','Gender']=1
        dm_split_add_satic=dm_split_add_satic[['subject_id',
                                'x0_study_id', 'x1_study_id',
                                'x0_dicom_id','x1_dicom_id',
                                'stay_id','ehr_len',
                                'left_time','right_time',
                                'mortality_inhospital',
                                            "Gender","Age"
                                ]].reset_index()
        
        
        dm_x0_chexpert=pd.merge(dm_split_add_satic,chexpert,
                left_on=['subject_id','x0_study_id'],right_on=['subject_id','study_id'],
                )

        dm_x0_x1_chexpert=pd.merge(dm_x0_chexpert,chexpert,
                left_on=['subject_id','x1_study_id'],right_on=['subject_id','study_id'],
                suffixes=['_l0','_l1'])
        
        # 0:uncertain, 1:negative, 2:positive, 3:not mentioned
        dm_x0_x1_chexpert=dm_x0_x1_chexpert.fillna(3)
        dm_x0_x1_chexpert[disease_cols_dm]=dm_x0_x1_chexpert[disease_cols_dm].replace(1,2).replace(0,1).replace(-1,0)
        print(f'length of {s} subset for ldm: {len(dm_x0_x1_chexpert)}')
        dm_x0_x1_chexpert.to_csv(os.path.join(args.output_csv_dir,f'{s}_dm_labels.csv'))

    
    ### Build  dataset for Autoencoder
    print("********************************")
    total_cxr_AP=cxr[(cxr['ViewPosition']=='AP')].dropna(subset=['ViewPosition'])[['subject_id','study_id','dicom_id']]
    
    for s in ['train','validate','test']:
        dm_subdataset=pd.read_csv(os.path.join(args.output_csv_dir,f'{s}_dm_labels.csv'))
        subjects=dm_subdataset.subject_id.unique()
        valid_cxr=total_cxr_AP[total_cxr_AP.subject_id.isin(subjects)]

        # merge with chexpert to get the chexpert labels
        df_merger_label=pd.merge(valid_cxr,chexpert,on=['subject_id','study_id'],how='left')
        print(f"length of {s} subset for autoencoder:",len(df_merger_label))
        # 0: uncertain, 1: negative, 2: positive 3: not mentioned
        df_merger_label=df_merger_label.fillna(3)
        df_merger_label[disease_cols]=df_merger_label[disease_cols].replace(1,2).replace(0,1).replace(-1,0)

        df_merger_label.to_csv(os.path.join(args.output_csv_dir,f'{s}_autoencoder_augmented_label.csv'))

    
    
    
   

