from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
import logging
from ehr_utils.preprocessing import Discretizer

random.seed(49297)
from tqdm import tqdm



# partition: train, val, test_upper, test_update
def prepare_dynamic_data(args, partition):
    # extract ehr data according to cxr interval
    cxr_df=pd.DataFrame('nameofcxr')
    part_cxr_df=cxr_df[cxr_df.split==partition].reset_index(drop=True)
    # sample_id(即为索引）, img_path,  split， subject_id, stay_id， y_l, y_d, y_intu,  s(考虑event和变量),
    # 变量信息就是从event里面来的，time series信息， 可以直接从time_series里面提取
    # 在df里面添加一列，放ts的文件路径(subject_id, stay_id, l_interval,r_interval)

    sample_id=0
    for index, row in part_cxr_df.iterrows():
        patient=row.subject_id
        stay_id=row.stay_id
        l_interval=row.l_interval
        r_interval=row.r_interval
        ts_df=pd.read_csv(os.path.join(args.root_path, patient,f'episode{stay_id}_timeseries.csv'))

        ts_df_interval=ts_df[(ts_df.Hours>= l_interval * 24 )&(ts_df.Hours <=r_interval * 24)]

        if len(ts_df_interval)==0:
            print(f'patinet:{patient},stay:{stay_id},day:{l_interval} to day:{r_interval} no ts ehr, was excluded.')
            continue

        # 将static变量拼接，并复制
        static_df=pd.read_csv(os.path.join(args.root_path, patient,f'episode{stay_id}.csv'))
        static_variable=["Gender","Age"]
        for v in static_variable:
            v_temp=static_df[v].iloc[0]
            ts_df_interval[v]=v_temp

        ts_path_interval=f'{patient}/{sample_id}_{patient}_{stay_id}_{l_interval}_{r_interval}.csv'
        ts_df_interval.to_csv(os.path.join(args.root_path,ts_path_interval),index=False)

        ts_df.loc[index,'sample_id']=sample_id
        ts_df.loc[index, 'ts_file'] = ts_path_interval

        sample_id+=1

    # 删除没有ehr的sample
    new_ts_df = ts_df.dropna(subset=['sample_id']).set_index('sample_id')
    print(f'missing ehr: {len(new_ts_df)-len(ts_df)}')
    print(f'{partition}: {len(new_ts_df)}')

    new_ts_df.to_csv(f'{partition}_samples.csv',index=True)


# 原来这个decompensation的代码是错的，因为每一个sample用的都是整个icu stay的ts

def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0,
                      eps=1e-6, future_time_interval=24.0):

    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("(length of stay is missing)", patient, ts_filename)
                    continue
                
                # import pdb; pdb.set_trace()
                
                stay = stays_df[stays_df.stay_id == label_df.iloc[0]['Icustay']]
                
                icustay = label_df['Icustay'].iloc[0]

                deathtime = stay['deathtime'].iloc[0]
                intime = stay['intime'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = 1e18
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("(no events in ICU) ", patient, ts_filename)
                    continue

                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)

                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                for t in sample_times:
                    if mortality == 0:
                        cur_mortality = 0
                    else:
                        # 判断sample的时候是否存活
                        cur_mortality = int(lived_time - t < future_time_interval)
                    xty_triples.append((output_ts_filename, t, icustay, cur_mortality))

    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,stay_id,y_true\n')
        for (x, t, icustay, y) in xty_triples:
            listfile.write('{},{:.6f},{},{:d}\n'.format(x, t, icustay, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
