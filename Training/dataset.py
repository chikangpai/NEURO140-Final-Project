from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold, StratifiedGroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from torch.utils.data import Dataset, Sampler,WeightedRandomSampler
import cv2
import matplotlib
import os
import sys
import random
import numpy as np
import pandas as pd
import glob
import torch
from os.path import join, basename, dirname
import yaml
from typing import List, Dict, Tuple, Literal
import copy
from util import list_files
import pickle

N_FOLDS = 4

class generateDataset:
    def __init__(self,
                 cancer,
                 fold,
                 task,
                 seed=24,
                 intDiagnosticSlide=0,
                 feature_type='tile',
                 strClinicalInformationPath="clinical_information",
                 strEmbeddingPath='',
                 geneType='',
                 geneName='',
                 **kwargs):
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        self.dfDistribution = None
        self.dictInformation = {}
        if self.cancer != None and self.fold != None:
            self.dfClinicalInformation = self.fClinicalInformation()
        else:
            self.dfClinicalInformation = None

    def fClinicalInformation(self):
        df = pd.DataFrame({})
        for c in self.cancer:
            part = pd.read_csv(glob.glob(join(
                'tcga_pan_cancer', f'{c.lower()}_tcga_pan_can_atlas_2018', 'clinical_data.tsv'))[0], sep='\t')
            df = pd.concat([df, part], ignore_index=True)
            label = pd.read_csv(glob.glob(join(
                'tcga_pan_cancer', f'{c.lower()}_tcga_pan_can_atlas_2018', '*', f'{self.geneType}_{self.geneName}*', '*.csv'))[0])
            label_filter = label[['Patient ID', 'Altered']]
            df = pd.merge(df, label_filter, on="Patient ID")
            df.rename(
                columns={'Patient ID': 'case_submitter_id', 'Altered': 'label'}, inplace=True)
            part['cancer'] = c
            df = pd.concat([df, part], ignore_index=True)
        return df

    def train_valid_test(self, split=1.0):
        if self.feature_type == 'slide':
            return self.train_valid_test_slidelevel(split)
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()

        # Load embedding paths
        if isinstance(self.strEmbeddingPath, dict):
            lsDownloadPath = []
            for cancer in self.cancer:
                if cancer == "coadread":
                    lsDownloadPath += list_files(self.strEmbeddingPath["COAD"], pattern='*.pt')
                    lsDownloadPath += list_files(self.strEmbeddingPath["READ"], pattern='*.pt')
                    continue
                path = self.strEmbeddingPath[cancer.upper()]
                lsDownloadPath += list_files(path, pattern='*.pt')
            lsDownloadPath = list(set(lsDownloadPath))
        else:
            lsDownloadPath = list_files(path, pattern='*.pt')

        # Process paths and create dataframe
        lsDownloadFoldID = [s.split('/')[-1][:-3] for s in lsDownloadPath]
        lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
        dfClinicalInformation = dfClinicalInformation[['case_submitter_id', 'label']]
        dfDownload = pd.DataFrame({
            'case_submitter_id': lsDownloadCaseSubmitterId,
            'folder_id': lsDownloadFoldID,
        })
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDownload, on="case_submitter_id")

        # Filter by slide type
        if (self.intDiagnosticSlide == 1):  # FFPE slide only
            dfClinicalInformation = dfClinicalInformation[[
                'DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)
        elif (self.intDiagnosticSlide == 0):  # frozen slide only
            dfClinicalInformation = dfClinicalInformation[[
                'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

        # Encode labels
        le = LabelEncoder()
        dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
            dfClinicalInformation.label.values)
        leLabel = le.classes_
        self.dictInformation["label"] = leLabel

        # Split data
        dfDummy = dfClinicalInformation.drop_duplicates(
            subset='case_submitter_id', ignore_index=True)
        dfDummy['fold'] = dfDummy['label'].tolist()
        dfDummy.fold = le.fit_transform(dfDummy.fold.values)
        foldNum = [0 for _ in range(int(len(dfDummy.index)))]

        if self.fold == 1:
            train, valitest = train_test_split(
                dfDummy, train_size=0.6, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
            vali, test = train_test_split(
                valitest, test_size=0.5, random_state=self.seed, shuffle=True, stratify=valitest['fold'].tolist())
            if split < 1.0:
                train, remainder = train_test_split(
                    train, train_size=split, random_state=self.seed, shuffle=True, stratify=train['fold'].tolist())
                bags = [list(train.index), list(vali.index), list(test.index)]
            else:
                bags = [list(train.index), list(vali.index), list(test.index)]

        # Assign fold numbers
        for fid, indices in enumerate(bags):
            for idx in indices:
                foldNum[idx] = fid
        dfDummy['fold'] = foldNum

        # Merge back with original data
        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDummy, on="case_submitter_id")

        # Add paths
        if isinstance(self.strEmbeddingPath, dict):
            df_pt = pd.DataFrame(
                {'folder_id': lsDownloadFoldID, 'path': lsDownloadPath})
            dfClinicalInformation = dfClinicalInformation.merge(
                df_pt, on='folder_id', how='inner')
        else:
            dfClinicalInformation['path'] = [
                f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]

        return dfClinicalInformation

    def train_valid_test_slidelevel(self, split=1.0):
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()

        with open(self.strEmbeddingPath, 'rb') as f:
            feature = pickle.load(f)
            lsDownloadFoldID = feature['filenames']
            lsDownloadPath = [self.strEmbeddingPath for _ in lsDownloadFoldID]

        lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
        dfClinicalInformation = dfClinicalInformation[['case_submitter_id', 'label']]
        lsDownloadPath = [s for s in lsDownloadPath if s.split(
            '/')[-1][:-3] in lsDownloadFoldID]
        dfDownload = pd.DataFrame({
            'case_submitter_id': lsDownloadCaseSubmitterId,
            'folder_id': lsDownloadFoldID,
        })
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDownload, on="case_submitter_id")

        if (self.intDiagnosticSlide == 1):  # FFPE slide only
            dfClinicalInformation = dfClinicalInformation[[
                'DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)
        elif (self.intDiagnosticSlide == 0):  # frozen slide only
            dfClinicalInformation = dfClinicalInformation[[
                'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

        le = LabelEncoder()
        dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
            dfClinicalInformation.label.values)
        leLabel = le.classes_
        self.dictInformation["label"] = leLabel

        dfDummy = dfClinicalInformation.drop_duplicates(
            subset='case_submitter_id', ignore_index=True)
        dfDummy['fold'] = dfDummy['label'].tolist()
        dfDummy.fold = le.fit_transform(dfDummy.fold.values)
        foldNum = [0 for _ in range(int(len(dfDummy.index)))]

        if self.fold == 1:
            train, valitest = train_test_split(
                dfDummy, train_size=0.6, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
            vali, test = train_test_split(
                valitest, test_size=0.5, random_state=self.seed, shuffle=True, stratify=valitest['fold'].tolist())
            if split < 1.0:
                train, remainder = train_test_split(
                    train, train_size=split, random_state=self.seed, shuffle=True, stratify=train['fold'].tolist())
                bags = [list(train.index), list(vali.index), list(test.index)]
            else:
                bags = [list(train.index), list(vali.index), list(test.index)]

        for fid, indices in enumerate(bags):
            for idx in indices:
                foldNum[idx] = fid
        dfDummy['fold'] = foldNum

        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDummy, on="case_submitter_id")

        if isinstance(self.strEmbeddingPath, dict):
            df_pt = pd.DataFrame(
                {'folder_id': lsDownloadFoldID, 'path': lsDownloadPath})
            dfClinicalInformation = dfClinicalInformation.merge(
                df_pt, on='folder_id', how='inner')
        else:
            dfClinicalInformation['path'] = [
                f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]

        return dfClinicalInformation

class CancerDataset(Dataset):
    def __init__(self, df, task, fold_idx=None, feature_type='tile',
                 split_type="kfold",
                 exp_idx=0,
                 max_train_tiles=None):
        PARTITION_TYPE_MAP = {0: 'training', 1: 'validation', 2: 'testing', None: 'all'}
        self.partition_type = PARTITION_TYPE_MAP[fold_idx]
        self.feature_type = feature_type
        self.df = df
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.exp_idx = exp_idx
        self.task = task
        self.max_train_tiles = max_train_tiles
        self.initialize_df()
        self.initialize_features()

    def initialize_df(self, df=None):
        if df is not None:
            self.df = df
        if self.fold_idx is None:
            print('No fold index is given. Will use the entire dataset')
            return
        if self.split_type == 'kfold':
            if self.fold_idx == 0:
                self.df = self.df[self.df['fold'].isin(
                    [(4-self.exp_idx) % 4, (4-self.exp_idx+1) % 4])].reset_index(drop=True)
            elif self.fold_idx == 1:
                self.df = self.df[self.df['fold'].isin(
                    [(4-self.exp_idx+2) % 4])].reset_index(drop=True)
            elif self.fold_idx == 2:
                self.df = self.df[self.df['fold'].isin(
                    [(4-self.exp_idx+3) % 4])].reset_index(drop=True)
            else:
                raise ValueError("Invalid fold index")
                
        elif self.split_type == 'vanilla':
            if self.fold_idx == 0:
                self.df = self.df[self.df['fold'].isin(
                    [0])].reset_index(drop=True)
            elif self.fold_idx == 1:
                self.df = self.df[self.df['fold'].isin(
                    [1])].reset_index(drop=True)
            elif self.fold_idx == 2:
                self.df = self.df[self.df['fold'].isin(
                    [2])].reset_index(drop=True)
            else:
                raise ValueError("Invalid fold index")
            
    def initialize_features(self):
        if self.feature_type == 'tile':
            return
        elif self.feature_type == 'slide':
            files = self.df['path'].unique()
            filenames = []
            features = []
            for file in files:
                with open(file, 'rb') as f:
                    feature = pickle.load(f)
                    filenames.extend(feature['filenames'])
                    features.append(feature['embeddings'])
            all_features = np.concatenate(features, axis=0).astype(np.float32)
            features = np.zeros((len(self.df), all_features.shape[1]))
            for idx, row in self.df.iterrows():
                filename = row['folder_id']
                idx_feature = filenames.index(filename)
                features[idx] = all_features[idx_feature]
            self.features = torch.from_numpy(features).float()
            
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        if self.feature_type == 'slide':
            sample = self.features[idx]
        else:
            sample = torch.load(row.path, map_location='cpu')

        if self.max_train_tiles and self.partition_type == 'training':
            if sample.shape[0] > self.max_train_tiles:
                idxs = np.random.choice(
                    sample.shape[0], self.max_train_tiles, replace=False)
                sample = sample[idxs, :]
        return sample, len(sample), row.label, row.case_submitter_id, row.folder_id

    def __len__(self):
        return len(self.df)

def get_datasets(df, task, split_type, exp_idx, 
                 feature_type='tile',
                 max_train_tiles=None):
    if split_type == 'kfold':
        train_ds = CancerDataset(
            df, task, 0, feature_type=feature_type,
            split_type=split_type, exp_idx=exp_idx, 
            max_train_tiles=max_train_tiles)
        val_ds = CancerDataset(
            df, task, 1, feature_type=feature_type,
            split_type=split_type, exp_idx=exp_idx)
        test_ds = CancerDataset(
            df, task, 2, feature_type=feature_type,
            split_type=split_type, exp_idx=exp_idx)
    elif split_type == 'vanilla':
        train_ds = CancerDataset(
            df, task, 0, feature_type=feature_type,
            split_type=split_type, max_train_tiles=max_train_tiles)
        val_ds = CancerDataset(df, task, 1, feature_type=feature_type,
                              split_type=split_type)
        test_ds = CancerDataset(df, task, 2, feature_type=feature_type,
                               split_type=split_type)

    return train_ds, val_ds, test_ds

def collate_fn(batch):
    samples, lengths, labels, case_submitter_id, folder_id = zip(*batch)
    max_len = max(lengths)
    padded_slides = []
    for i in range(0, len(samples)):
        pad = (0, 0, 0, max_len-lengths[i])
        padded_slide = torch.nn.functional.pad(samples[i], pad)
        padded_slides.append(padded_slide)
    padded_slides = torch.stack(padded_slides)
    return padded_slides, lengths, torch.tensor(labels), list(case_submitter_id), list(folder_id)

