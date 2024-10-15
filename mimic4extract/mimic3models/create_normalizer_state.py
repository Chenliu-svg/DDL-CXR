from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
# from mimic4extract.mimic3models.preprocessing import Discretizer, Normalizer
from ehr_utils.preprocessing import Discretizer, Normalizer
import os
import argparse

Variable_headers=[]

def main():
    parser = argparse.ArgumentParser(description='Script for creating a normalizer state - a file which stores the '
                                                 'means and standard deviations of columns of the output of a '
                                                 'discretizer, which are later used to standardize the input of '
                                                 'neural models.')
    # parser.add_argument('--task', type=str, required=True,
    #                     choices=['ihm', 'decomp', 'los', 'pheno', 'multi'])
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="Rate of the re-sampling to discretize time-series.")
    parser.add_argument('--impute_strategy', type=str, default='previous',
                        choices=['zero', 'next', 'previous', 'normal_value'],
                        help='Strategy for imputing missing values.')
    parser.add_argument('--start_time', type=str,default='zero', choices=['zero', 'relative'],
                        help='Specifies the start time of discretization. Zero means to use the beginning of '
                             'the ICU stay. Relative means to use the time of the first ICU event')
    parser.add_argument('--store_masks', dest='store_masks', action='store_true',
                        help='Store masks that specify observed/imputed values.')
    parser.add_argument('--no-masks', dest='store_masks', action='store_false',
                        help='Do not store that specify specifying observed/imputed values.')
    parser.add_argument('--n_samples', type=int, default=-1, help='How many samples to use to estimates means and '
                        'standard deviations. Set -1 to use all training samples.')
    parser.add_argument('--output_dir', type=str, help='Directory where the output file will be saved.',
                        default='.')
    parser.add_argument('--data_dir', type=str, default='/home/lpp/DiffusionModel/latent-diffusion/data/root', help='Path to the task data.')
    parser.set_defaults(store_masks=True)

    args = parser.parse_args()
    print(args)



    # create the discretizer
    discretizer = Discretizer(timestep=args.timestep,
                              store_masks=True,
                              impute_strategy=args.impute_strategy,
                              start_time=args.start_time)

    cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61]
    # create the normalizer
    normalizer = Normalizer()

    # read all examples and store the state of the normalizer
    train_samples=pd.read_csv(os.path.join(args.data_dir,'train_pred.csv'))
    n_samples = len(train_samples)

    for index, row in train_samples.iterrows():


        if index % 1000 == 0:
            print('Processed {} / {} samples'.format(index, n_samples), end='\r')

        # ret = pd.read_csv(os.path.join(args.data_dir,row.ts_file)).fillna("").values
        subject_id =train_samples.loc[index, 'subject_id']
        stay_id =train_samples.loc[index, 'stay_id']

        age=train_samples.loc[index, 'Age']
        gender=train_samples.loc[index, 'Gender']

        ts_df = pd.read_csv(os.path.join(args.data_dir, str(subject_id), f'episode{stay_id}_timeseries.csv'))
        ts_interval = ts_df[(ts_df.Hours >= 0) & (
                ts_df.Hours <= 48)].reset_index(drop=True).fillna("").values
        ## add static
        ts_interval=np.c_[ts_interval,np.ones(ts_interval.shape[0])*gender]
        ts_interval = np.c_[ts_interval,np.ones(ts_interval.shape[0])*age]

        data, new_header = discretizer.transform(ts_interval,end=48)

        # new_header_ls=new_header.split(',')
        # with open('../../header.txt', 'w') as file:
        #     file.write(new_header)
        # fields= [new_header_ls.index(k) for k,v in discretizer._is_categorical_channel.items() if not v ]
        # print(fields)
        # exit(0)
        normalizer._feed_data(data)
    print('\n')

    file_name = 'ts:{:.2f}_impute:{}_start:{}_masks:{}_n:{}.normalizer'.format(
         args.timestep, args.impute_strategy, args.start_time, args.store_masks, n_samples)
    file_name = os.path.join(args.output_dir, file_name)
    print('Saving the state in {} ...'.format(file_name))
    normalizer._save_params(file_name)
    normalizer.load_params(file_name)

    print(normalizer._means[cont_channels])

    print(normalizer._stds[cont_channels])


# def main():
#     parser = argparse.ArgumentParser(description='Script for creating a normalizer state - a file which stores the '
#                                                  'means and standard deviations of columns of the output of a '
#                                                  'discretizer, which are later used to standardize the input of '
#                                                  'neural models.')
#     parser.add_argument('--task', type=str, required=True,
#                         choices=['ihm', 'decomp', 'los', 'pheno', 'multi'])
#     parser.add_argument('--timestep', type=float, default=1.0,
#                         help="Rate of the re-sampling to discretize time-series.")
#     parser.add_argument('--impute_strategy', type=str, default='previous',
#                         choices=['zero', 'next', 'previous', 'normal_value'],
#                         help='Strategy for imputing missing values.')
#     parser.add_argument('--start_time', type=str, choices=['zero', 'relative'],
#                         help='Specifies the start time of discretization. Zero means to use the beginning of '
#                              'the ICU stay. Relative means to use the time of the first ICU event')
#     parser.add_argument('--store_masks', dest='store_masks', action='store_true',
#                         help='Store masks that specify observed/imputed values.')
#     parser.add_argument('--no-masks', dest='store_masks', action='store_false',
#                         help='Do not store that specify specifying observed/imputed values.')
#     parser.add_argument('--n_samples', type=int, default=-1, help='How many samples to use to estimates means and '
#                         'standard deviations. Set -1 to use all training samples.')
#     parser.add_argument('--output_dir', type=str, help='Directory where the output file will be saved.',
#                         default='.')
#     parser.add_argument('--data', type=str, required=True, help='Path to the task data.')
#     parser.set_defaults(store_masks=True)
#
#     args = parser.parse_args()
#     print(args)
#
#     # create the reader
#     reader = None
#     dataset_dir = os.path.join(args.data, 'train')
#     if args.task == 'ihm':
#         reader = InHospitalMortalityReader(dataset_dir=dataset_dir, period_length=48.0)
#     if args.task == 'decomp':
#         reader = DecompensationReader(dataset_dir=dataset_dir)
#     if args.task == 'los':
#         reader = LengthOfStayReader(dataset_dir=dataset_dir)
#     if args.task == 'pheno':
#         reader = PhenotypingReader(dataset_dir=dataset_dir)
#     if args.task == 'multi':
#         reader = MultitaskReader(dataset_dir=dataset_dir)
#
#     # create the discretizer
#     discretizer = Discretizer(timestep=args.timestep,
#                               store_masks=args.store_masks,
#                               impute_strategy=args.impute_strategy,
#                               start_time=args.start_time)
#     discretizer_header = reader.read_example(0)['header']
#     continuous_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
#
#     # create the normalizer
#     normalizer = Normalizer(fields=continuous_channels)
#
#     # read all examples and store the state of the normalizer
#     n_samples = args.n_samples
#     if n_samples == -1:
#         n_samples = reader.get_number_of_examples()
#
#     for i in range(n_samples):
#         if i % 1000 == 0:
#             print('Processed {} / {} samples'.format(i, n_samples), end='\r')
#         ret = reader.read_example(i)
#         data, new_header = discretizer.transform(ret['X'], end=ret['t'])
#         normalizer._feed_data(data)
#     print('\n')
#
#     file_name = '{}_ts:{:.2f}_impute:{}_start:{}_masks:{}_n:{}.normalizer'.format(
#         args.task, args.timestep, args.impute_strategy, args.start_time, args.store_masks, n_samples)
#     file_name = os.path.join(args.output_dir, file_name)
#     print('Saving the state in {} ...'.format(file_name))
#     normalizer._save_params(file_name)


if __name__ == '__main__':
    main()
