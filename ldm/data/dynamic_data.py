import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

from ehr_utils.preprocessing import Discretizer, Normalizer
import h5py
import pickle


PAD = 0

FINGDINGS=['Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Enlarged Cardiomediastinum',
 'Fracture',
 'Lung Lesion',
 'Lung Opacity',
 'No Finding',
 'Pleural Effusion',
 'Pleural Other',
 'Pneumonia',
 'Pneumothorax',
 'Support Devices']

CLASSES_DM=[i+'_l0' for i in FINGDINGS]+[i+'_l1' for i in FINGDINGS]
X1_CLASSES_DM=[i+'_l1' for i in FINGDINGS]

class CollateFunc():
    def __init__(self, dim=0, dataset_mode='na'):
        self.dim = dim
        self.dataset_mode = dataset_mode


    
    def pad_collate_ldm_label(self, batch):

        x0 = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        x1 = torch.stack(list(map(lambda x: x[2], batch)), dim=0)
        ehr_y = torch.stack(list(map(lambda x: x[3], batch)), dim=0)
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        padded_ehr = list(map(lambda x: self.pad_tensor(x[0], pad=max_len, dim=self.dim), batch))
        # ehr
        ehr = torch.stack(list(map(lambda x: x[0], padded_ehr)), dim=0)
        masks = torch.stack(list(map(lambda x: x[1], padded_ehr)), dim=0)

        sample_id = torch.tensor(list(map(lambda x: x[4], batch)))


        return ehr, masks, x0, x1, ehr_y, sample_id



    def pad_collate_ldm_gen(self,batch):
        # sample_id, x_0, ehr
        sample_id = torch.tensor(list(map(lambda x: x[0], batch)))
        x0 = torch.stack(list(map(lambda x: x[1], batch)), dim=0)

        max_len = max(map(lambda x: x[2].shape[self.dim], batch))
        batch = list(map(lambda x: self.pad_tensor(x[2], pad=max_len, dim=self.dim), batch))
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        masks = torch.stack(list(map(lambda x: x[1], batch)), dim=0)

        return  xs, masks, x0, sample_id



    @staticmethod
    def pad_tensor(vec, pad, dim):
        if isinstance(vec,np.ndarray):
            vec = torch.from_numpy(vec)
        origin_shape = vec.shape
        pad_size = list(origin_shape)
        pad_size[dim] = pad - vec.size(dim)
        padded_vec = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
        mask = torch.cat([torch.ones(origin_shape[dim] + 1), torch.zeros(pad - origin_shape[dim])], dim=dim)
        return padded_vec, mask.to(torch.bool)

    def __call__(self, batch):

        if self.dataset_mode=='generate_z1':
            return self.pad_collate_ldm_gen(batch)
        elif self.dataset_mode == 'ldm_label':
            return self.pad_collate_ldm_label(batch)
    
        else:
            return batch

class VAE_Dataset(Dataset):
    """
    Prepare splitted data (cxr, chexpert label) for VAE
    
    Args:
        partition: train, validate, split
        mimic_cxr_jpg_dir: path to mimic-cxr-jpg files
        metadata_path: path to the {partition}_autoencoder_augmented_label.csv, where informations(dicom_id,chexpert labels) of the cxr is.
    """
    def __init__(self,
                 partition,
                 mimic_cxr_jpg_dir='/root/autodl-tmp/dynamic_cxr',  
                 metadata_path='./data/',
                 resize=256,
                 crop=224,
                 
    ):
        self.imag_path=mimic_cxr_jpg_dir
        
        self.data_df = pd.read_csv(os.path.join(metadata_path, f'{partition}_autoencoder_augmented_label.csv'))

        # replace chexpert labels to [0,1]: 0: not known, uncertain, negative  1:positive
        self.data_df[FINGDINGS]=self.data_df[FINGDINGS].replace(1,0).replace(3,0).replace(2,1)
        for col in FINGDINGS:
            assert all(self.data_df[col].isin([0, 1])), f"Column '{col}' should only contain 0 and 1."

        self.partition = partition
        
        self.resize=resize
        self.crop=crop
        train_transforms, test_transforms=self.get_transforms()
        if partition=='train':
            self.transform= transforms.Compose(train_transforms)
        else:
            self.transform= transforms.Compose(test_transforms)

       
    def __len__(self):
        return len(self.data_df)
    
    def get_transforms(self):
        normalize = transforms.Normalize(mean=0.5, std=0.5)

        train_transforms = []
        train_transforms.append(transforms.Resize(self.resize))
        train_transforms.append(transforms.RandomCrop(self.crop))
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)      


        test_transforms = []
        test_transforms.append(transforms.Resize(self.resize))
        test_transforms.append(transforms.CenterCrop(self.crop))
        test_transforms.append(transforms.ToTensor())
        test_transforms.append(normalize)

        return train_transforms, test_transforms


    def __getitem__(self, index):

        # get image
        subject_id = self.data_df.loc[index, 'subject_id']
        study_id = self.data_df.loc[index, 'study_id']
        dicom_id = self.data_df.loc[index, 'dicom_id']
        x_0_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        # x_0_path=f'{dicom_id}.jpg'
        abs_x_0_path = os.path.join(self.imag_path, x_0_path)
        
        x_0 = Image.open(abs_x_0_path).convert('RGB')
        
        x_0 = self.transform(x_0)

        # get labels
        label=torch.tensor(list(self.data_df.loc[index,FINGDINGS].values))

        return (x_0,label)

class EHR_Pickler():
    """
    params:
        stage: ldm, generation or prediction
        metadata_path:  where the meta csv files are
        mimic_iv_subjects_dir: The path to the processed mimic-iv subjects files, where the  `{subject_id}/episode{stay_id}_timeseries.csv` is.
    """

    def __init__(self,
                 
                 stage,
                 metadata_path='data/',
                 mimic_iv_subjects_dir='data/mimic_iv_subjects_dir/' ,
                 timestep=1.00,
                 normalizer_state="ldm/data/normalizers/dm_normalizer",
                 
        
                 ):
        self.stage=stage
        
        self.start_time = "zero" if stage=='prediction' else "relative"
        discretizer = Discretizer(timestep=float(timestep),
                                  store_masks=True,
                                  impute_strategy='previous',
                                  start_time=self.start_time)
        
        self.cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61]
        normalizer = Normalizer(fields=self.cont_channels)  # choose here which columns to standardize
        normalizer.load_params(normalizer_state)

        self.discretizer = discretizer
        self.normalizer = normalizer
        
        self.discret_end = 48 if stage=='prediction' else None 
        self.csv_suffix='pred' if stage=='prediction' else 'dm_labels' 
    
        self.metadata_path = metadata_path
        self.mimic_iv_subjects_dir=mimic_iv_subjects_dir
        
       
 

    def _ehr_pickle_dm(self):
        for s in ['train','validate','test']:
            data_df = pd.read_csv(os.path.join(self.metadata_path, f'{s}_{self.csv_suffix}.csv'))
            ts_dict={}
            for _, row in data_df.iterrows():
                sample_id=str(row['index'])
                subject_id = row['subject_id']
                stay_id = row['stay_id']
                left_time = row['left_time']
                right_time = row[ 'right_time']

                age = row['Age']
                gender = row['Gender']

                ts_df = pd.read_csv(os.path.join(self.mimic_iv_subjects_dir, str(subject_id), f'episode{stay_id}_timeseries.csv'))
                ts_interval = ts_df[(ts_df.Hours >= left_time) & (
                        ts_df.Hours <= right_time)].reset_index(drop=True).fillna("").values

                ## add static
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * gender]
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * age]

                ## discrete and normalize
                ts_data = self.discretizer.transform(ts_interval)[0]

                ehr = self.normalizer.transform(ts_data)
                ts_dict[sample_id]=ehr
            
            
            with open(os.path.join(self.metadata_path,f'{s}_ehr_dm.pkl'), 'wb') as f:
                pickle.dump(ts_dict, f)

            print(f'Loading ehr file ')
            with open(os.path.join(self.metadata_path,f'{s}_ehr_dm.pkl'), 'rb') as f:
                ts = pickle.load(f)
            print(list(ts.values())[0].shape)
            print('done!')
            print(len(ts_dict))
            
    def _ehr_pickle_gen(self):
        for s in ['train','validate','test']:
            data_df = pd.read_csv(os.path.join(self.metadata_path, f'{s}_{self.csv_suffix}.csv'))
            ts_dict={}
            for _, row in data_df.iterrows():
                sample_id=str(row['index'])
                subject_id = row['subject_id']
                stay_id = row['stay_id']
                cha=row['cha']
                age = row['Age']
                gender = row['Gender']
                
                
                ts_df = pd.read_csv(os.path.join(self.mimic_iv_subjects_dir, str(subject_id), f'episode{stay_id}_timeseries.csv'))

                if cha>=12:
                    ts_interval = ts_df[(ts_df.Hours >= (48-cha)) & (
                        ts_df.Hours <= 48)].reset_index(drop=True).fillna("").values
                else:
                    ts_interval = ts_df[(ts_df.Hours >= 36) & (
                        ts_df.Hours <= 48)].reset_index(drop=True).fillna("").values
                ## add static
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * gender]
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * age]
                ## discrete and normalize
                ts_data = self.discretizer.transform(ts_interval, end=self.discret_end)[0]
                ehr = torch.from_numpy(self.normalizer.transform(ts_data))
                
                ts_dict[sample_id]=ehr
            
            with open(os.path.join(self.metadata_path,f'{s}_ehr_gen.pkl'), 'wb') as f:
                pickle.dump(ts_dict, f)
                
            print(f'Loading ehr file ')
            with open(os.path.join(self.metadata_path,f'{s}_ehr_gen.pkl'), 'rb') as f:
                ts = pickle.load(f)
            print(list(ts.values())[0].shape)
            print('done!')
            print(len(ts_dict))
       

    def _ehr_pickle_pred(self):
        for s in ['train','validate','test']:
            data_df = pd.read_csv(os.path.join(self.metadata_path, f'{s}_{self.csv_suffix}.csv'))
            ts_dict={}
            for _, row in data_df.iterrows():
                sample_id=str(row['index'])
                subject_id = row['subject_id']
                stay_id = row['stay_id']
                cha=row['cha']
                age = row['Age']
                gender = row['Gender']
                
                
                ts_df = pd.read_csv(os.path.join(self.mimic_iv_subjects_dir, str(subject_id), f'episode{stay_id}_timeseries.csv'))

                ts_interval = ts_df[(ts_df.Hours >= 0) & (
                        ts_df.Hours <= 48)].reset_index(drop=True).fillna("").values
                ## add static
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * gender]
                ts_interval = np.c_[ts_interval, np.ones(ts_interval.shape[0]) * age]
                ## discrete and normalize
                ts_data = self.discretizer.transform(ts_interval, end=self.discret_end)[0]
                ehr = torch.from_numpy(self.normalizer.transform(ts_data))
                
                ts_dict[sample_id]=ehr
            
            with open(os.path.join(self.metadata_path,f'{s}_ehr_pred.pkl'), 'wb') as f:
                pickle.dump(ts_dict, f)
                
            print(f'Loading ehr file ')
            with open(os.path.join(self.metadata_path,f'{s}_ehr_pred.pkl'), 'rb') as f:
                ts = pickle.load(f)
            print(list(ts.values())[0].shape)
            print('done!')
            print(len(ts_dict))
                
    def pickle_ehr(self):
        if self.stage=='prediction':
            return self._ehr_pickle_pred()
        elif self.stage=='generation':
            return self._ehr_pickle_gen()
        else:
            return self._ehr_pickle_dm() 

class LDM_Dataset(Dataset):
    """
    params:
        partition: train, validate, test
        mimic_cxr_jpg_dir: image path to mimic cxr jpg files
        metadata_path: where the meta csv files, ehr pikle files are
        
    """

    def __init__(self,
                 partition,
                 mimic_cxr_jpg_dir,
                 metadata_path='./data/',
                 resize=224,
                 crop=224,
                 ):
        
        super().__init__()
        self.mimic_cxr_jpg_dir = mimic_cxr_jpg_dir
        
        self.metadata_path=metadata_path
        self.data_df = pd.read_csv(os.path.join(self.metadata_path, f'{partition}_dm_labels.csv'))
        
    
        # replace chexpert labels
        self.data_df[CLASSES_DM]=self.data_df[CLASSES_DM].replace(1,0).replace(3,0).replace(2,1)
        for col in CLASSES_DM:
            assert all(self.data_df[col].isin([0, 1])), f"Column '{col}' should only contain 0 and 1."


        self.partition = partition
        
        train_transforms, test_transforms=self.get_transforms(resize,crop)
        if partition=='train':
            self.transform= transforms.Compose(train_transforms)
        else:
            self.transform= transforms.Compose(test_transforms)
        
        # ehr 
        print(f'Loading EHR data from {partition}_ehr_dm.pkl')
        with open(os.path.join(self.metadata_path, f'{partition}_ehr_dm.pkl'), 'rb') as f:
            self.processed_ehr = pickle.load(f)
        
    def __len__(self):
        return len(self.data_df)

 


    def __getitem__(self, index):
          
        subject_id = self.data_df.loc[index, 'subject_id']
        sample_id = self.data_df.loc[index, 'index']
        ehr=self.processed_ehr[str(sample_id)]
        
        study_id = self.data_df.loc[index, 'x0_study_id']
        dicom_id = self.data_df.loc[index, 'x0_dicom_id']
        x_0_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        # x_0_path = f'{dicom_id}.jpg'
        abs_x_0_path = os.path.join(self.mimic_cxr_jpg_dir, x_0_path)
        
        x_0 = Image.open(abs_x_0_path).convert('RGB')
        x_0 = self.transform(x_0)

        study_id = self.data_df.loc[index, 'x1_study_id']
        dicom_id = self.data_df.loc[index, 'x1_dicom_id']
        x_1_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        # x_1_path = f'{dicom_id}.jpg'
        abs_x_1_path = os.path.join(self.mimic_cxr_jpg_dir, x_1_path)
        
        x_1 = Image.open(abs_x_1_path).convert('RGB')
        x_1 = self.transform(x_1)

        # get labels
        ehr_y=torch.tensor(list(self.data_df.loc[index,X1_CLASSES_DM].values))

        return (ehr, x_0, x_1, ehr_y, sample_id)
    


    def get_transforms(self,resize,crop):
       
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        train_transforms = []
      
        train_transforms.append(transforms.Resize(resize))
        train_transforms.append(transforms.CenterCrop(crop))
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)      
        test_transforms = []
        
        test_transforms.append(transforms.Resize(resize))
        test_transforms.append(transforms.CenterCrop(crop))
        test_transforms.append(transforms.ToTensor())
        test_transforms.append(normalize)

        return train_transforms, test_transforms



class GenerateZ1(Dataset):
    """
    params:
        partition: train, validate, test 
        mimic_cxr_jpg_dir: image path to mimic cxr jpg files
        metadata_path: root to label files, ehr pickle files ,output the latent cxr
    """

    def __init__(self,
                 partition,
                 mimic_cxr_jpg_dir,
                 metadata_path='/root/autodl-tmp/data//',
                 resize=224,
                 crop=224,
                 ):
        super().__init__()
        
        self.metadata_path=metadata_path
        # meta files
        self.data_df = pd.read_csv(os.path.join(self.metadata_path, f'{partition}_pred.csv'))
        self.mimic_cxr_jpg_dir= mimic_cxr_jpg_dir
        self.transform = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop), transforms.ToTensor(),
                          transforms.Normalize(mean=0.5, std=0.5)])

        # ehr 
        print(f'Loading EHR data from {partition}_ehr_gen.pkl')
        with open(os.path.join(self.metadata_path, f'{partition}_ehr_gen.pkl'), 'rb') as f:
            self.processed_ehr = pickle.load(f)


    def __len__(self):
        return len(self.data_df)


    def __getitem__(self, index):

        sample_id = self.data_df.loc[index, 'index']

        ehr=self.processed_ehr[str(sample_id)]


        subject_id = self.data_df.loc[index, 'subject_id']
        study_id = self.data_df.loc[index, 'study_id']
        dicom_id = self.data_df.loc[index, 'dicom_id']

        x_0_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        
        # x_0_path = f'{dicom_id}.jpg'
        abs_x_0_path = os.path.join(self.mimic_cxr_jpg_dir, x_0_path)
        x_0 = Image.open(abs_x_0_path).convert('RGB')
        
        x_0 = self.transform(x_0)


        return sample_id, x_0, ehr


class Generate_CXR(Dataset):
    """
    params:
        partition: train, validate, test
        mimic_cxr_jpg_dir: image path to mimic cxr jpg files
        metadata_path: where the meta csv files, ehr pikle files are
        
    """

    def __init__(self,
                 mimic_cxr_jpg_dir,
                 partition='test',
                 metadata_path='./data/',
                 resize=224,
                 crop=224,
                 ):
        
        super().__init__()
        self.mimic_cxr_jpg_dir = mimic_cxr_jpg_dir
        
        self.metadata_path=metadata_path
        self.data_df = pd.read_csv(os.path.join(self.metadata_path, f'{partition}_dm_labels.csv'))
        
        # fo test
        # self.data_df=self.data_df[self.data_df['subject_id'].astype(str).str.startswith('10')]
        # print(len(self.data_df))
        
        test_transforms=self.get_transforms(resize,crop)
        self.transform= transforms.Compose(test_transforms)
        
        # ehr 
        print(f'Loading EHR data from {partition}_ehr_dm.pkl')
        with open(os.path.join(self.metadata_path, f'{partition}_ehr_dm.pkl'), 'rb') as f:
            self.processed_ehr = pickle.load(f)
        
    def __len__(self):
        return len(self.data_df)



    def __getitem__(self, index):
        # index=2
        subject_id = self.data_df.loc[index, 'subject_id']
        sample_id = self.data_df.loc[index, 'index']
        ehr=self.processed_ehr[str(sample_id)]
        
        study_id = self.data_df.loc[index, 'x0_study_id']
        dicom_id = self.data_df.loc[index, 'x0_dicom_id']
        x_0_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        # x_0_path = f'{dicom_id}.jpg'
        abs_x_0_path = os.path.join(self.mimic_cxr_jpg_dir, x_0_path)
        
        x_0 = Image.open(abs_x_0_path).convert('RGB')
        x_0 = self.transform(x_0)

        study_id = self.data_df.loc[index, 'x1_study_id']
        dicom_id = self.data_df.loc[index, 'x1_dicom_id']
        x_1_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        # x_1_path = f'{dicom_id}.jpg'
        abs_x_1_path = os.path.join(self.mimic_cxr_jpg_dir, x_1_path)
        
        x_1 = Image.open(abs_x_1_path).convert('RGB')
        x_1 = self.transform(x_1)

        # get labels
        ehr_y=torch.tensor(list(self.data_df.loc[index,X1_CLASSES_DM].values))

        return (ehr, x_0, x_1, ehr_y, sample_id)
    


    def get_transforms(self,resize,crop):
       
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        test_transforms = []
        test_transforms.append(transforms.Resize(resize))
        test_transforms.append(transforms.CenterCrop(crop))
        test_transforms.append(transforms.ToTensor())
        test_transforms.append(normalize)

        return test_transforms



        
class PredictDataset(Dataset):

    def __init__(self,
                 partition,
                 task,
                 
                 metadata_path='/root/autodl-tmp/data/',
                 mimic_cxr_jpg_dir='/root/autodl-tmp/dynamic_new/dynamic_cxr/',
                 
                 ):
        super().__init__()
        # meta (labels)
        self.task=task
        self.partition=partition
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transforms = []
        train_transforms.append(transforms.Resize(256))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
        train_transforms.append(transforms.CenterCrop(224))
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)      

        test_transforms = []
        test_transforms.append(transforms.Resize(256))


        test_transforms.append(transforms.CenterCrop(224))

        test_transforms.append(transforms.ToTensor())
        test_transforms.append(normalize) 

        self.transform={'train':transforms.Compose(train_transforms),'validate':transforms.Compose(test_transforms),'test':transforms.Compose(test_transforms)}
        
    
        self.data_root=metadata_path
        self.data_df = pd.read_csv(os.path.join(self.data_root, f'{partition}_pred.csv'))
        
        self.gen_h5file = h5py.File(os.path.join(metadata_path,f'{self.partition}_z1_for_pred.h5'), 'r')

        self.mimic_cxr_jpg_dir=mimic_cxr_jpg_dir
        
       
        print(f'Loading EHR data from {partition}_ehr_pred.pkl')
        with open(os.path.join(self.data_root, f'{partition}_ehr_pred.pkl'), 'rb') as f:
            self.processed_ehr = pickle.load(f)
            

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        if self.task=='mortality':
            y = torch.tensor(self.data_df.loc[index, 'mortality_inhospital'])
        
        else:
            assert self.task=='phenotype'
            y = torch.from_numpy(self.data_df.loc[index].values[-25:].astype(float))
        sample_id = self.data_df.loc[index, 'index']
 
        ehr=self.processed_ehr[str(sample_id)]
        
        subject_id = self.data_df.loc[index, 'subject_id']
        study_id = self.data_df.loc[index, 'study_id']
        dicom_id = self.data_df.loc[index, 'dicom_id']
        x_0_path = f'p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg'
        z1=torch.from_numpy(self.gen_h5file[(f'{sample_id}')][()])
        # x_0_path = f'{dicom_id}.jpg'
        abs_x_0_path = os.path.join(self.mimic_cxr_jpg_dir, x_0_path)
        
        

        x_0 = Image.open(abs_x_0_path).convert('RGB')
        x_0 = self.transform[self.partition](x_0)
        
        return (ehr, y, x_0, sample_id,z1)   
        
