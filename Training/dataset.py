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

from stable_sample_reweighting import column_wise_resampling, decorrelation
from collections import Counter

TISSUE_TYPE_COL = 'tissue_type'
TUMOR_STAGE_COL = 'ajcc_pathologic_stage'
## hard-coded dataset-specific arguments (could be moved to a separate yaml file in the future)


class generateTCGADataSet():
    def __init__(self,
                 cancer,
                 sensitive,
                 fold,
                 task,
                 seed=24,
                 intDiagnosticSlide=0,
                 feature_type='tile',
                 strClinicalInformationPath="clinical_information",
                 age_col='Diagnosis Age',
                 strEmbeddingPath='',
                 geneType='',
                 geneName='',
                 label_map_yaml=None,
                 sensitive_attr_map_yaml=None,
                 **kwargs):
        # if(cancer != None and len(cancer) < 1): raise ValueError(f'cancer less than 1.')
        # add all the parameters to the class
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        # set some custom parameters
        self.intTumor = 1
        self.sort = False
        self.dfDistribution = None
        self.dfRemoveDupDistribution = None
        self.dictInformation = {}
        # if self.cancer != None and self.sensitive != None and self.fold != None:
        if self.cancer != None and self.fold != None:
            self.dfClinicalInformation = self.fClinicalInformation()
        else:
            self.dfClinicalInformation = None

    def replace_na(self, df, na_kws=["'--", 'not reported', np.nan]):
        # replace na strings to pd.na
        for kw in na_kws:
            df = df.replace(kw, pd.NA)
        return df

    def fClinicalInformation(self):
        df = pd.DataFrame({})
        for c in self.cancer:
            if self.task == 4:      # genetic classification
                # part = pd.read_csv(glob.glob(f'tcga_pan_cancer/{c}_tcga_pan_can_atlas_2018/clinical_data.tsv')[0], sep='\t')
                part = pd.read_csv(glob.glob(join(
                    'tcga_pan_cancer', f'{c.lower()}_tcga_pan_can_atlas_2018', 'clinical_data.tsv'))[0], sep='\t')
                df = pd.concat([df, part], ignore_index=True)
                # label = pd.read_csv(glob.glob(f'tcga_pan_cancer/{c}_tcga_pan_can_atlas_2018/*/{self.geneType}_{self.geneName}*/*.csv')[0])
                label = pd.read_csv(glob.glob(join(
                    'tcga_pan_cancer', f'{c.lower()}_tcga_pan_can_atlas_2018', '*', f'{self.geneType}_{self.geneName}*', '*.csv'))[0])
                label_filter = label[['Patient ID', 'Altered']]
                df = pd.merge(df, label_filter, on="Patient ID")
                df.rename(
                    columns={'Patient ID': 'case_submitter_id', 'Altered': 'label'}, inplace=True)
            elif self.task in [1, 2]:      # cancer classification or detection
                # part = pd.read_csv(glob.glob(f'clinical_information/*{c}*.csv')[0])
                csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
                    glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
                assert len(csvs) > 0, f"No clinical information file found for {c}"
                part = pd.read_csv(csvs[0])
            elif self.task == 3:    # survival analysis
                csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
                    glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
                assert len(csvs) > 0, f"No clinical information file found for {c}"
                part = pd.read_csv(csvs[0])
            else:
                csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
                    glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
                assert len(csvs) > 0, f"No clinical information file found for {c}"
                part = pd.read_csv(csvs[0])
            ## add cancer column
            part['cancer'] = c
            df = pd.concat([df, part], ignore_index=True)
        # replace na strings to pd.na
        df = self.replace_na(df)
        # if sensitive attribute is age, convert it age group
        if self.sensitive is not None:
            sensitive_col = list(self.sensitive.keys())[0]
            if sensitive_col == 'age':
                df = self.mapAge(df)
        ##
        return df

    def fReduceDataFrame(self, df):
        ## get sensitive attribute (if None, use label as dummy sensitive attribute)


        if self.task == 3:
            dfClinicalInformation = df.copy()
            mask = (df['days_to_death'] ==
                    '\'--') & (df['days_to_last_follow_up'] == '\'--')

            dfClinicalInformation = dfClinicalInformation[~mask].reset_index(
                drop=True)
            dfClinicalInformation['event'] = dfClinicalInformation['days_to_death'].apply(
                lambda x: 1 if x != '\'--' else 0)  # 1: death 0: alive
            # print(dfClinicalInformation['event'].value_counts())
            # print("inside", dfClinicalInformation)
            mask2 = dfClinicalInformation['days_to_death'] != '\'--'
            dfClinicalInformation.loc[mask2,
                                      'T'] = dfClinicalInformation.loc[mask2, 'days_to_death']
            dfClinicalInformation.loc[~mask2,
                                      'T'] = dfClinicalInformation.loc[~mask2, 'days_to_last_follow_up']

            stages = ['Stage I', 'Stage IA', 'Stage IB',
                      'Stage II', 'Stage IIA', 'Stage IIB', 'Stage IIC']
            mask3 = dfClinicalInformation[TUMOR_STAGE_COL].isin(stages)
            dfClinicalInformation = dfClinicalInformation.loc[mask3]
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'event'
            dfClinicalInformation = dfClinicalInformation[['case_submitter_id', sensitive_col, 'T', 'event', TUMOR_STAGE_COL]]

            if self.sensitive_attr_map_yaml is not None:
                df = self.mapSensitive(df)
            dfClinicalInformation.columns = [
                'case_submitter_id', 'sensitive', 'T', 'event', 'stage']
            # drop rows without label
            dfClinicalInformation = dfClinicalInformation.dropna(subset=[
                                                                 'event', 'T'])

            return dfClinicalInformation

        elif self.task == 4:
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'label' # dummy sensitive attribute

            df = df[['case_submitter_id', sensitive_col, 'label']]

        elif self.task in [1, 2]:
            if (len(self.cancer) == 1):
                sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'primary_diagnosis' # dummy sensitive attribute
                df = df[['case_submitter_id', sensitive_col, 'primary_diagnosis']]
            else:
                sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'cancer' # dummy sensitive attribute
                df = df[['case_submitter_id', sensitive_col, 'cancer']]
        elif type(self.task) == str:
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else self.task
            df = df[['case_submitter_id', sensitive_col, self.task]]
            if self.label_map_yaml is not None:
                df = self.mapLabel(df)

        if self.sensitive_attr_map_yaml is not None:
            df = self.mapSensitive(df)
        df.columns = ['case_submitter_id', 'sensitive', 'label']
        # drop rows without label
        df = df.dropna(subset=['label'])

        return df

    def fTransLabel(self, df):
        pass

    def fTransSensitive(self, df):
        if self.sensitive is None: # if no sensitive attribute is given, return the original dataframe
            return df
        substrings = self.sensitive[list(self.sensitive.keys())[0]]
        df['sensitive'] = df['sensitive'].fillna("None")
        df = df[[any(x in y for x in substrings)
                 for y in df['sensitive'].tolist()]]
        assert len(df) > 0, "No data found for the given sensitive groups"
        return df

    def updateDataFrame(self):
        self.dfClinicalInformation = self.fClinicalInformation()

    def getDistribution(self):
        return self.dfDistribution

    def printDistribution(self):
        print(self.dfDistribution)

    def getRemoveDupDistribution(self):
        return self.dfRemoveDupDistribution

    def getInformation(self):
        return self.dictInformation

    def mapLabel(self, df):
        assert type(
            self.task) == str, "mapLabel is only for custom tasks (string format)"
        # check if label_map_yaml is provided
        if self.label_map_yaml is None:
            Warning(
                "label_map_yaml is not provided. Assuming the labels are already encoded.")
            return df
        # load label_map_yaml
        with open(self.label_map_yaml, 'r') as stream:
            try:
                label_map = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # check if the given task is in the label_map_yaml
        if self.task not in label_map.keys():
            print(
                f"Task {self.task} is not in the label_map_yaml. Assuming the labels are already encoded.")
            return df
        # map labels and drop NaN
        label_map = label_map[self.task]
        # for debugging purpose, list the labels that are not in the label_map
        not_in_map = [l for l in df[self.task].unique()
                      if l not in label_map.keys()]
        if len(not_in_map) > 0:
            print(f"Labels not in the label_map: {not_in_map}")
        df[self.task] = df[self.task].map(label_map)
        df = df.dropna(subset=[self.task])
        return df

    def mapAge(self, df):
        def number_or_na(x):
            try:
                out = float(x)
            except:
                out = pd.NA
            return out
        # process age column
        # df_temp =
        df[self.age_col] = df[self.age_col].apply(number_or_na)
        df = df.loc[df[self.age_col].notna()].reset_index(drop=True)
        median_age = df[self.age_col].median()
        df['age'] = df[self.age_col].apply(
            lambda x: 'old' if x > median_age else 'young')
        df = df.dropna(subset=['age'])
        return df

    def mapSensitive(self, df):
        if self.sensitive is None: # if no sensitive attribute is given, return the original dataframe
            return df
        sensitive_col = list(self.sensitive.keys())[0]
        # if sensitive_col == 'age':
        #     return self.mapAge(df)
        # for foolproofing, check if the sensitive_attr_map_yaml is provided
        if self.sensitive_attr_map_yaml is None:
            Warning(
                "sensitive_attr_map_yaml is not provided. Assuming the sensitive attributes are already encoded.")
            return df
        # load the sensitive_attr_map_yaml
        with open(self.sensitive_attr_map_yaml, 'r') as stream:
            try:
                sensitive_attr_map = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # check if the sensitive attribute is in the sensitive_attr_map_yaml
        if sensitive_col not in sensitive_attr_map.keys():
            print(
                f"Sensitive attribute {sensitive_col} is not in the sensitive_attr_map_yaml. Assuming the sensitive attributes are already encoded.")
            return df
        # map the sensitive attributes and drop NaN
        sensitive_attr_map = sensitive_attr_map[sensitive_col]
        # for debugging purpose, list the sensitive attributes that are not in the sensitive_attr_map
        not_in_map = [l for l in df[sensitive_col].unique(
        ) if l not in sensitive_attr_map.keys()]
        if len(not_in_map) > 0:
            print(
                f"Sensitive attributes not in the sensitive_attr_map: {not_in_map}")
        df[sensitive_col] = df[sensitive_col].map(sensitive_attr_map)
        df = df.dropna(subset=[sensitive_col])
        return df
    

    
    def train_valid_test(self, split=1.0):
        if self.feature_type == 'slide':
            return self.train_valid_test_slidelevel(split)
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()

        if isinstance(self.strEmbeddingPath, dict):
            # if embeddings are stored in separate folders
            lsDownloadPath = []
            for cancer in self.cancer:
                if cancer == "coadread":  # handle task4 naming convention
                    lsDownloadPath += list_files(self.strEmbeddingPath["COAD"], pattern='*.pt')
                    lsDownloadPath += list_files(self.strEmbeddingPath["READ"], pattern='*.pt')
                    continue
                path = self.strEmbeddingPath[cancer.upper()]
                # lsDownloadPath += glob.glob(f'{path}/*.pt')
                lsDownloadPath += list_files(path, pattern='*.pt')
            lsDownloadPath = list(set(lsDownloadPath))  # remove duplicates
        # elif isinstance(self.strEmbeddingPath, list):
            # if embeddings are stored in a list
            # lsDownloadPath = list_files(path, pattern='*.pt')
        else:
            lsDownloadPath = list_files(path, pattern='*.pt')

        lsDownloadFoldID = [s.split('/')[-1][:-3] for s in lsDownloadPath]

        # task: cancer classification
        if self.task == 1:

            if (self.intTumor == 1):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] == '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] == '0' for s in lsDownloadPath]].tolist()
            elif (self.intTumor == 0):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] != '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] != '0' for s in lsDownloadPath]].tolist()

            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="case_submitter_id")

            if (self.intDiagnosticSlide == 1):  # FFPE slide only
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)
            elif (self.intDiagnosticSlide == 0):  # frozen slide only
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

            if (len(self.cancer) == 1):
                if (self.cancer[0] == 'BRCA'):
                    positive = ['Infiltrating duct and lobular carcinoma', 'Infiltrating duct carcinoma',
                                'Infiltrating duct mixed with other types of carcinoma']
                    negative = [
                        'Infiltrating lobular mixed with other types of carcinoma', 'Lobular carcinoma']
                    bags = [positive, negative]
                    labels = dfClinicalInformation['label'].tolist()
                    replace = []
                    for lb in labels:
                        flag = True
                        for idx, conds in enumerate(bags):
                            for cond in conds:
                                if cond in lb:
                                    replace += [idx]
                                    flag = False
                                    break
                                if (flag == False):
                                    break
                            if (flag == False):
                                break
                        if (flag == True):
                            replace += [2]
                    dfClinicalInformation['label'] = replace
                    dfClinicalInformation = dfClinicalInformation[dfClinicalInformation['label'] != 2]
                    leLabel = ['IDC+', 'ILC']
            else:
                # Order the cancer types by what is in the --cancer argument
                for i, cancer in enumerate(self.cancer):
                    dfClinicalInformation.loc[dfClinicalInformation['label'] == cancer, 'label'] = i
                leLabel = self.cancer

            self.dictInformation['label'] = leLabel

        # task: tumor detection
        elif self.task == 2:
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="case_submitter_id")

            if (self.intDiagnosticSlide == 1):
                raise ValueError(
                    "Tumor detection task is not implemented for FFPE slides")
            elif (self.intDiagnosticSlide == 0):
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

            project_ids = dfClinicalInformation['folder_id'].apply(  #label from folder id 
                lambda x: x[13]).tolist()
            project_ids = [1 if x == '0' else 0 for x in project_ids]  #exchange 0, 1 -> 0(no tumor), 1(tumor)

            dfClinicalInformation.loc[:, 'label'] = project_ids
            self.dictInformation['label'] = ['normal','tumor']


        # task: survival analysis
        elif self.task == 3:
            raise ValueError("Survival analysis task is not refactored yet for TCGA dataset")
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            # print("after", dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
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
            dfClinicalInformation.loc[:, 'event'] = le.fit_transform(
                dfClinicalInformation.event.values)
            leLabel = le.classes_
            self.dictInformation['event'] = leLabel

            scaler = MinMaxScaler()
            # apply standardization before MinMaxScaler
            dfClinicalInformation.loc[:, 'T'] = (dfClinicalInformation['T'].astype(
                float)-dfClinicalInformation['T'].astype(float).mean())/dfClinicalInformation['T'].astype(float).std()
            dfClinicalInformation.loc[:, 'T'] = scaler.fit_transform(
                dfClinicalInformation['T'].values.reshape(-1, 1))

            stage_map = {
                'Stage I': 0,
                'Stage IA': 0,
                'Stage IB': 0,
                'Stage II': 1,
                'Stage IIA': 1,
                'Stage IIB': 1,
                'Stage IIC': 1
            }
            dfClinicalInformation.loc[:, 'stage'] = dfClinicalInformation['stage'].map(
                stage_map)

        elif self.task == 4:
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
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
        ############################################
        # custom tasks in string format.
        # in this case, the task name specifies the label column in the dataset.
        elif type(self.task) == str:
            if (self.intTumor == 1):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] == '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] == '0' for s in lsDownloadPath]].tolist()
            elif (self.intTumor == 0):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] != '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] != '0' for s in lsDownloadPath]].tolist()

            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            # dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
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
            dfClinicalInformation.loc[:, 'label_old'] = dfClinicalInformation.label.copy(
            )
            dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
                dfClinicalInformation.label.values)
            leLabel = le.classes_
            self.dictInformation['label'] = leLabel

        # df.columns = ['case_submitter_id', 'sensitive', 'label','folder_id']
        print('======   Data Summary (slide level)   ======')
        
        print(dfClinicalInformation[['sensitive',
              'label']].value_counts().sort_index())
        print('============================================')
        
        le = LabelEncoder()
        dfClinicalInformation = self.fTransSensitive(
            dfClinicalInformation).reset_index(drop=True)
        dfClinicalInformation.sensitive = le.fit_transform(
            dfClinicalInformation.sensitive.values)
        leSensitive = le.classes_
        self.dictInformation['sensitive'] = leSensitive
        print(f'Number of unique patients: {dfClinicalInformation["case_submitter_id"].nunique()}')

        if self.task != 2:
            # if not tumor detection, assume one label per patient and drop duplicates
            # then stratify by sensitive attribute and label
            dfDummy = dfClinicalInformation.drop_duplicates(
                subset='case_submitter_id', ignore_index=True)
            if self.task == 3:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['event'])).tolist()
            else:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['label'])).tolist()
            dfDummy.fold = le.fit_transform(dfDummy.fold.values)
            foldNum = [0 for _ in range(int(len(dfDummy.index)))]
        else:

            df = dfClinicalInformation
            counter = Counter(df['sensitive'])

            # Step 1: Assign folds to (label=0, sensitive=minor) cases using KFold
            subset_1 = df[(df['label'] == 0) & (df['sensitive'] == min(counter))].copy()
            n_folds = 4
            kf1 = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            subset_1['fold'] = ''

            for fold, (_, test_idx) in enumerate(kf1.split(subset_1)):
                subset_1.iloc[test_idx, subset_1.columns.get_loc('fold')] = fold


            # Step 2: Assign folds to (label=0, sensitive=minor) cases using KFold
            subset_2 = df[(df['label'] == 1) & (df['sensitive'] == min(counter))].copy()
            kf2 = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            subset_2['fold'] = ''

            for fold, (_, test_idx) in enumerate(kf2.split(subset_2)):
                subset_2.iloc[test_idx, subset_2.columns.get_loc('fold')] = fold


            # Step 2: Assign folds to remaining cases using Stratified K-Fold
            remaining = df[~df.index.isin(subset_1.index)].copy()
            remaining = remaining[~remaining.index.isin(subset_2.index)].copy()


            # Combine 'label' and 'sensitive' into a single stratification column
            # Create a combined column for stratification
            remaining['stratify_col'] = remaining['label'].astype(str) + "_" + remaining['sensitive'].astype(str)

            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            remaining['fold'] = -1

            for fold, (_, test_idx) in enumerate(sgkf.split(remaining, remaining['label'], groups=remaining['case_submitter_id'])):
                remaining.iloc[test_idx, remaining.columns.get_loc('fold')] = fold

            # Combine both subsets back
            dfDummy = pd.concat([subset_1, remaining]).drop(columns=['stratify_col'])


          


            # raise NotImplementedError(
            #     "We need a custom stratification for tumor detection task because slides from the same patient can have different labels.\n Sophie please help implement this.")
        if self.task != 2:
            if self.fold == 1:
                train, valitest = train_test_split(
                    dfDummy, train_size=0.6, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
                vali, test = train_test_split(
                    valitest, test_size=0.5, random_state=self.seed, shuffle=True, stratify=valitest['fold'].tolist())
                # split training data
                if split < 1.0:
                    train, remainder = train_test_split(
                        train, train_size=split, random_state=self.seed, shuffle=True, stratify=train['fold'].tolist())
                    bags = [list(train.index), list(vali.index), list(test.index)]
                    print("The length of training set: ", len(remainder))
                else:
                    bags = [list(train.index), list(vali.index), list(test.index)]

            elif self.fold == 2:
                ab, cd = train_test_split(
                    dfDummy, train_size=0.5, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
                a, b = train_test_split(
                    ab, test_size=0.5, random_state=self.seed, shuffle=True, stratify=ab['fold'].tolist())
                c, d = train_test_split(
                    cd, train_size=0.5, random_state=self.seed, shuffle=True, stratify=cd['fold'].tolist())
                bags = [list(a.index), list(b.index), list(c.index), list(d.index)]

            for fid, indices in enumerate(bags):
                for idx in indices:
                    foldNum[idx] = fid
            dfDummy['fold'] = foldNum

        # if self.task == 2:
        #     print('sophie')

        if self.task == 3:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'label', 'sensitive']).size()

        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDummy, on="case_submitter_id")

        if isinstance(self.strEmbeddingPath, dict):
            # if embeddings span multiple folders
            # TODO remove np.array(lsDownloadFoldID)[[s[13] != '0' for s in lsDownloadFoldID]].tolist()
            df_pt = pd.DataFrame(
                {'folder_id': lsDownloadFoldID, 'path': lsDownloadPath})
            dfClinicalInformation = dfClinicalInformation.merge(
                df_pt, on='folder_id', how='inner')
        else:
            dfClinicalInformation['path'] = [
                f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]

        if self.task == 3:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'label', 'sensitive']).size()
        self.printDistribution()
        return dfClinicalInformation


    
    def train_valid_test_slidelevel(self, split=1.0):
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()

            # for slide-level features, all features are in a single pkl
        with open(self.strEmbeddingPath, 'rb') as f:
            feature = pickle.load(f)
            lsDownloadFoldID = feature['filenames']
            lsDownloadPath = [self.strEmbeddingPath for _ in lsDownloadFoldID ] # for backward compatibility

        # lsDownloadFoldID = [s.split('/')[-1][:-3] for s in lsDownloadPath]

        # task: cancer classification
        if self.task == 1:

            if (self.intTumor == 1):
                idx = [i for i, x in enumerate(lsDownloadFoldID) if x[13] == '0']
                lsDownloadFoldID = [lsDownloadFoldID[i] for i in idx]
                lsDownloadPath = [lsDownloadPath[i] for i in idx]
            elif (self.intTumor == 0):
                idx = [i for i, x in enumerate(lsDownloadFoldID) if x[13] != '0']
                lsDownloadFoldID = [lsDownloadFoldID[i] for i in idx]
                lsDownloadPath = [lsDownloadPath[i] for i in idx]

            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="case_submitter_id")

            if (self.intDiagnosticSlide == 1):  # FFPE slide only
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)
            elif (self.intDiagnosticSlide == 0):  # frozen slide only
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

            if (len(self.cancer) == 1):
                if (self.cancer[0] == 'BRCA'):
                    positive = ['Infiltrating duct and lobular carcinoma', 'Infiltrating duct carcinoma',
                                'Infiltrating duct mixed with other types of carcinoma']
                    negative = [
                        'Infiltrating lobular mixed with other types of carcinoma', 'Lobular carcinoma']
                    bags = [positive, negative]
                    labels = dfClinicalInformation['label'].tolist()
                    replace = []
                    for lb in labels:
                        flag = True
                        for idx, conds in enumerate(bags):
                            for cond in conds:
                                if cond in lb:
                                    replace += [idx]
                                    flag = False
                                    break
                                if (flag == False):
                                    break
                            if (flag == False):
                                break
                        if (flag == True):
                            replace += [2]
                    dfClinicalInformation['label'] = replace
                    dfClinicalInformation = dfClinicalInformation[dfClinicalInformation['label'] != 2]
                    leLabel = ['IDC+', 'ILC']
            else:
                # Order the cancer types by what is in the --cancer argument
                for i, cancer in enumerate(self.cancer):
                    dfClinicalInformation.loc[dfClinicalInformation['label'] == cancer, 'label'] = i
                leLabel = self.cancer

            self.dictInformation['label'] = leLabel

        # task: tumor detection
        elif self.task == 2:
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="case_submitter_id")

            if (self.intDiagnosticSlide == 1):
                raise ValueError(
                    "Tumor detection task is not implemented for FFPE slides")
            elif (self.intDiagnosticSlide == 0):
                dfClinicalInformation = dfClinicalInformation[[
                    'DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop=True)

            project_ids = dfClinicalInformation['folder_id'].apply(  #label from folder id 
                lambda x: x[13]).tolist()
            project_ids = [1 if x == '0' else 0 for x in project_ids]  #exchange 0, 1 -> 0(no tumor), 1(tumor)

            dfClinicalInformation.loc[:, 'label'] = project_ids
            self.dictInformation['label'] = ['normal','tumor']


        # task: survival analysis
        elif self.task == 3:
            raise ValueError("Survival analysis task is not refactored yet for TCGA dataset")
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            # print("after", dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
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
            dfClinicalInformation.loc[:, 'event'] = le.fit_transform(
                dfClinicalInformation.event.values)
            leLabel = le.classes_
            self.dictInformation['event'] = leLabel

            scaler = MinMaxScaler()
            # apply standardization before MinMaxScaler
            dfClinicalInformation.loc[:, 'T'] = (dfClinicalInformation['T'].astype(
                float)-dfClinicalInformation['T'].astype(float).mean())/dfClinicalInformation['T'].astype(float).std()
            dfClinicalInformation.loc[:, 'T'] = scaler.fit_transform(
                dfClinicalInformation['T'].values.reshape(-1, 1))

            stage_map = {
                'Stage I': 0,
                'Stage IA': 0,
                'Stage IB': 0,
                'Stage II': 1,
                'Stage IIA': 1,
                'Stage IIB': 1,
                'Stage IIC': 1
            }
            dfClinicalInformation.loc[:, 'stage'] = dfClinicalInformation['stage'].map(
                stage_map)

        elif self.task == 4:
            # raise NotImplementedError("Not implemented yet")
            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(
                dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
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
        ############################################
        # custom tasks in string format.
        # in this case, the task name specifies the label column in the dataset.
        elif type(self.task) == str:
            if (self.intTumor == 1):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] == '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] == '0' for s in lsDownloadPath]].tolist()
            elif (self.intTumor == 0):
                lsDownloadFoldID = np.array(lsDownloadFoldID)[
                    [s[13] != '0' for s in lsDownloadFoldID]].tolist()
                lsDownloadPath = np.array(lsDownloadPath)[
                    [s.split("/")[-1][13] != '0' for s in lsDownloadPath]].tolist()

            lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            # dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
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
            dfClinicalInformation.loc[:, 'label_old'] = dfClinicalInformation.label.copy(
            )
            dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
                dfClinicalInformation.label.values)
            leLabel = le.classes_
            self.dictInformation['label'] = leLabel

        # df.columns = ['case_submitter_id', 'sensitive', 'label','folder_id']
        print('======   Data Summary (slide level)   ======')
        
        print(dfClinicalInformation[['sensitive',
              'label']].value_counts().sort_index())
        print('============================================')
        
        le = LabelEncoder()
        dfClinicalInformation = self.fTransSensitive(
            dfClinicalInformation).reset_index(drop=True)
        dfClinicalInformation.sensitive = le.fit_transform(
            dfClinicalInformation.sensitive.values)
        leSensitive = le.classes_
        self.dictInformation['sensitive'] = leSensitive
        print(f'Number of unique patients: {dfClinicalInformation["case_submitter_id"].nunique()}')

        if self.task != 2:
            # if not tumor detection, assume one label per patient and drop duplicates
            # then stratify by sensitive attribute and label
            dfDummy = dfClinicalInformation.drop_duplicates(
                subset='case_submitter_id', ignore_index=True)
            if self.task == 3:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['event'])).tolist()
            else:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['label'])).tolist()
            dfDummy.fold = le.fit_transform(dfDummy.fold.values)
            foldNum = [0 for _ in range(int(len(dfDummy.index)))]
        else:
# <<<<<<< Sophie
            df = dfClinicalInformation
            counter = Counter(df['sensitive'])

            # Step 1: Assign folds to (label=0, sensitive=minor) cases using KFold
            subset_1 = df[(df['label'] == 0) & (df['sensitive'] == min(counter))].copy()
            n_folds = 4
            kf1 = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            subset_1['fold'] = ''

            for fold, (_, test_idx) in enumerate(kf1.split(subset_1)):
                subset_1.iloc[test_idx, subset_1.columns.get_loc('fold')] = fold


            # Step 2: Assign folds to (label=0, sensitive=minor) cases using KFold
            subset_2 = df[(df['label'] == 1) & (df['sensitive'] == min(counter))].copy()
            kf2 = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            subset_2['fold'] = ''

            for fold, (_, test_idx) in enumerate(kf2.split(subset_2)):
                subset_2.iloc[test_idx, subset_2.columns.get_loc('fold')] = fold


            # Step 2: Assign folds to remaining cases using Stratified K-Fold
            remaining = df[~df.index.isin(subset_1.index)].copy()
            remaining = remaining[~remaining.index.isin(subset_2.index)].copy()


            # Combine 'label' and 'sensitive' into a single stratification column
            # Create a combined column for stratification
            remaining['stratify_col'] = remaining['label'].astype(str) + "_" + remaining['sensitive'].astype(str)

            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            remaining['fold'] = -1

            for fold, (_, test_idx) in enumerate(sgkf.split(remaining, remaining['label'], groups=remaining['case_submitter_id'])):
                remaining.iloc[test_idx, remaining.columns.get_loc('fold')] = fold

            # Combine both subsets back
            dfDummy = pd.concat([subset_1, remaining]).drop(columns=['stratify_col'])

          


            # raise NotImplementedError(
            #     "We need a custom stratification for tumor detection task because slides from the same patient can have different labels.\n Sophie please help implement this.")
        if self.task != 2:
            if self.fold == 1:
                train, valitest = train_test_split(
                    dfDummy, train_size=0.6, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
                vali, test = train_test_split(
                    valitest, test_size=0.5, random_state=self.seed, shuffle=True, stratify=valitest['fold'].tolist())
                # split training data
                if split < 1.0:
                    train, remainder = train_test_split(
                        train, train_size=split, random_state=self.seed, shuffle=True, stratify=train['fold'].tolist())
                    bags = [list(train.index), list(vali.index), list(test.index)]
                    print("The length of training set: ", len(remainder))
                else:
                    bags = [list(train.index), list(vali.index), list(test.index)]

            elif self.fold == 2:
                ab, cd = train_test_split(
                    dfDummy, train_size=0.5, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
                a, b = train_test_split(
                    ab, test_size=0.5, random_state=self.seed, shuffle=True, stratify=ab['fold'].tolist())
                c, d = train_test_split(
                    cd, train_size=0.5, random_state=self.seed, shuffle=True, stratify=cd['fold'].tolist())
                bags = [list(a.index), list(b.index), list(c.index), list(d.index)]

            for fid, indices in enumerate(bags):
                for idx in indices:
                    foldNum[idx] = fid
            dfDummy['fold'] = foldNum

        # if self.task == 2:
        #     print('sophie')

        if self.task == 3:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'label', 'sensitive']).size()

        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDummy, on="case_submitter_id")

        if isinstance(self.strEmbeddingPath, dict):
            # if embeddings span multiple folders
            # TODO remove np.array(lsDownloadFoldID)[[s[13] != '0' for s in lsDownloadFoldID]].tolist()
            df_pt = pd.DataFrame(
                {'folder_id': lsDownloadFoldID, 'path': lsDownloadPath})
            dfClinicalInformation = dfClinicalInformation.merge(
                df_pt, on='folder_id', how='inner')
        else:
            dfClinicalInformation['path'] = [
                f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]

        if self.task == 3:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'label', 'sensitive']).size()
        self.printDistribution()
        return dfClinicalInformation



class generateExternalDataSet(generateTCGADataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fClinicalInformation(self):
        df = pd.DataFrame({})
        for c in self.cancer:
            if self.task == 4:      # genetic classification
                # part = pd.read_csv(glob.glob(f'tcga_pan_cancer/{c}_tcga_pan_can_atlas_2018/clinical_data.tsv')[0], sep='\t')
                # part = pd.read_csv(glob.glob(join(
                #     'tcga_pan_cancer', f'{c}_tcga_pan_can_atlas_2018', 'clinical_data.tsv'))[0], sep='\t')
                csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
                    glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
                assert len(csvs) > 0, f"No clinical information file found for {c}"
                part = pd.read_csv(csvs[0])                
            else:
                # part = pd.read_csv(glob.glob(f'clinical_information/*{c}*.csv')[0])
                csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
                    glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
                assert len(csvs) > 0, f"No clinical information file found for {c}"
                part = pd.read_csv(csvs[0])
            # elif self.task == 3:    # survival analysis
            #     csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
            #         glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
            #     assert len(csvs) > 0, f"No clinical information file found for {c}"
            #     part = pd.read_csv(csvs[0])
            # else:
            #     csvs = glob.glob(join(self.strClinicalInformationPath, f'*{c.lower()}*.csv')) + \
            #         glob.glob(join(self.strClinicalInformationPath, f'*{c.upper()}*.csv'))
            #     assert len(csvs) > 0, f"No clinical information file found for {c}"
            #     part = pd.read_csv(csvs[0])
            ## add cancer column
            part['cancer'] = c
            df = pd.concat([df, part], ignore_index=True)
            
        if self.task == 4:
            gene_col = f'{self.geneName}{self.mutation_kw}'
            df = df.rename(
                columns={'Patient ID': 'case_submitter_id', gene_col: 'label'})
        # replace na strings to pd.na
        df = self.replace_na(df)
        # if sensitive attribute is age, convert it age group
        if self.sensitive is not None:
            sensitive_col = list(self.sensitive.keys())[0]
            if sensitive_col == 'age':
                df = self.mapAge(df)
        ##
        return df

    def fTransSensitive(self, df, exact=False):
        if self.sensitive is None: # if no sensitive attribute is given, return the original dataframe
            return df
        substrings = self.sensitive[list(self.sensitive.keys())[0]]
        # drop na
        df.dropna(subset=['sensitive'], inplace=True)
        # drop sensitive attributes that are not in the list
        if exact:
            df = df[[any(x == y for x in substrings)
                     for y in df['sensitive'].tolist()]]
        else:
            df = df[[any(x in y for x in substrings)
                     for y in df['sensitive'].tolist()]]
        return df

    def fReduceDataFrame(self, df):
        # if not tumor detection, only keep tumor slides
        if self.task != 2:
            if TISSUE_TYPE_COL in df.columns:
                df = df[df[TISSUE_TYPE_COL] == 'Tumor']
            else:
                Warning(
                    f'Tissue type column not found in clinical information file. Assuming all slides are tumor slides.')

        if self.task == 3:  # survival analysis
            dfClinicalInformation = df.copy()
            mask = (df['days_to_death'] ==
                    '\'--') & (df['days_to_last_follow_up'] == '\'--')
            dfClinicalInformation = dfClinicalInformation[~mask].reset_index(
                drop=True)
            dfClinicalInformation['event'] = dfClinicalInformation['days_to_death'].apply(
                lambda x: 1 if x != '\'--' else 0)  # 1: death 0: alive
            # print(dfClinicalInformation['event'].value_counts())
            mask2 = dfClinicalInformation['days_to_death'] != '\'--'
            dfClinicalInformation.loc[mask2,
                                      'T'] = dfClinicalInformation.loc[mask2, 'days_to_death']
            dfClinicalInformation.loc[~mask2,
                                      'T'] = dfClinicalInformation.loc[~mask2, 'days_to_last_follow_up']

            stages = ['Stage I', 'Stage IA', 'Stage IB',
                      'Stage II', 'Stage IIA', 'Stage IIB', 'Stage IIC']
            mask3 = dfClinicalInformation[TUMOR_STAGE_COL].isin(stages)
            dfClinicalInformation = dfClinicalInformation.loc[mask3]
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'event'

            dfClinicalInformation = dfClinicalInformation[['case_submitter_id', sensitive_col, 'T', 'event', TUMOR_STAGE_COL]]

            if self.sensitive_attr_map_yaml is not None:
                df = self.mapSensitive(df)

            dfClinicalInformation.columns = [
                'case_submitter_id', 'sensitive', 'T', 'event', 'stage']
            return dfClinicalInformation

        elif self.task == 1:  # cancer classification
            if (len(self.cancer) == 1):
                # df = df[['case_submitter_id', list(self.sensitive.keys())[0], 'primary_diagnosis','folder_id']]
                raise NotImplementedError(
                    """
                    Only one cancer type is given for the cancer type classification
                    Are you doing CANCER SUBTYPE classification? If so, please do the following instead:
                    - Prepare the clinical information file for each subtype specifically (e.g. BRCA_ductal)
                    - Add the subtype to the dictionary "strEmbeddingPath" (different subtype can share the same feature folder. They just need to have their own entry in the dictionary)
                    """
                )
            else:
                sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'project_id'

                # df = df[['case_submitter_id', sensitive_col, 'project_id', 'folder_id']]
                df = df[['case_submitter_id', sensitive_col, 'cancer', 'folder_id']]
                df.columns = ['case_submitter_id',
                              'sensitive', 'label', 'folder_id']

        elif self.task == 2:  # tumor detection
            # use tissue type as label
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else TISSUE_TYPE_COL

            df = df[['case_submitter_id', sensitive_col, TISSUE_TYPE_COL, 'folder_id']]
            df.columns = ['case_submitter_id',
                          'sensitive', 'label', 'folder_id']
        elif self.task == 4:
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else 'label' # dummy sensitive attribute

            df = df[['case_submitter_id', sensitive_col, 'label','folder_id']]


        elif type(self.task) == str:
            sensitive_col = list(self.sensitive.keys())[0] if self.sensitive is not None else self.task

            df = df[['case_submitter_id', sensitive_col, self.task, 'folder_id']]
            if self.label_map_yaml is not None:
                df = self.mapLabel(df)

        if self.sensitive_attr_map_yaml is not None:
            df = self.mapSensitive(df)

        df.columns = ['case_submitter_id', 'sensitive', 'label', 'folder_id']
        # convert folder_id to string
        df['folder_id'] = df['folder_id'].astype(str)

        print('=======\tData Summary\t========')
        print(df[['sensitive', 'label']].value_counts())
        print('====================================')

        return df

    def train_valid_test(self, split=1.0):
        if self.dfClinicalInformation is None:
            self.updateDataFrame()
        dfClinicalInformation = self.dfClinicalInformation.copy()
        if isinstance(self.strEmbeddingPath, dict):
            # if embeddings are stored in separate folders
            lsDownloadPath = []
            for cancer in self.cancer:
                path = self.strEmbeddingPath[cancer]
                cancer_files = glob.glob(f'{path}/*.pt')
                assert len(cancer_files) > 0, f"""No embeddings found for {cancer} ({path})"""
                lsDownloadPath += cancer_files
            lsDownloadPath = list(set(lsDownloadPath))  # remove duplicates
        else:
            lsDownloadPath = glob.glob(f'{self.strEmbeddingPath}/*.pt')

        lsDownloadFoldID = [s.split('/')[-1][:-3] for s in lsDownloadPath]

        # task: cancer classification
        if self.task == 1:
            # Disable filtering for tumor slides. DFCI contains only tumor slides.
            # if(self.intTumor == 0):
            #     lsDownloadFoldID = np.array(lsDownloadFoldID)[[s[13] == '0' for s in lsDownloadFoldID]].tolist()
            # elif(self.intTumor == 1):
            #     lsDownloadFoldID = np.array(lsDownloadFoldID)[[s[13] != '0' for s in lsDownloadFoldID]].tolist()

            # lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            # dfClinicalInformation = self.fReduceDataFrame(
            #     dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation)
            dfDownload = pd.DataFrame({
                # 'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            ##
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="folder_id", how='inner')
            # Disable filtering for sample type. DFCI contains only FFPE slides
            # if(self.intDiagnosticSlide == 0):
            #     dfClinicalInformation = dfClinicalInformation[['DX' in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop = True)
            # elif(self.intDiagnosticSlide == 1):
            #     dfClinicalInformation = dfClinicalInformation[['DX' not in s[20:22] for s in dfClinicalInformation['folder_id'].tolist()]].reset_index(drop = True)

            le = LabelEncoder()
            if (len(self.cancer) == 1):
                if (self.cancer[0] == 'BRCA'):
                    positive = ['Infiltrating duct and lobular carcinoma', 'Infiltrating duct carcinoma',
                                'Infiltrating duct mixed with other types of carcinoma']
                    negative = [
                        'Infiltrating lobular mixed with other types of carcinoma', 'Lobular carcinoma']
                    bags = [positive, negative]
                    labels = dfClinicalInformation['label'].tolist()
                    replace = []
                    for lb in labels:
                        flag = True
                        for idx, conds in enumerate(bags):
                            for cond in conds:
                                if cond in lb:
                                    replace += [idx]
                                    flag = False
                                    break
                                if (flag == False):
                                    break
                            if (flag == False):
                                break
                        if (flag == True):
                            replace += [2]
                    dfClinicalInformation['label'] = replace
                    dfClinicalInformation = dfClinicalInformation[dfClinicalInformation['label'] != 2]
                    leLabel = bags
            else:
                # Order the cancer types by what is in the --cancer argument
                for i, cancer in enumerate(self.cancer):
                    dfClinicalInformation.loc[dfClinicalInformation['label'] == cancer, 'label'] = i
                leLabel = self.cancer
            self.dictInformation['label'] = leLabel

        # task: tumor detection
        elif self.task == 2:
            # dfClinicalInformation = self.fReduceDataFrame(
            #     dfClinicalInformation.drop_duplicates(subset='case_submitter_id', ignore_index=True))
            # dfDownload = pd.DataFrame({
            #     # 'case_submitter_id': lsDownloadCaseSubmitterId,
            #     'folder_id': lsDownloadFoldID
            # })
            # dfClinicalInformation = pd.merge(
            #     dfClinicalInformation, dfDownload, on="folder_id", how='inner')
            
            dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation)
            dfDownload = pd.DataFrame({
                # 'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            ##
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="folder_id", how='inner')


            dfClinicalInformation['label'] = dfClinicalInformation['label'].apply(
                lambda x: 0 if x.lower() == 'normal' else 1)

            self.dictInformation['label'] = ['Normal', 'Tumor']

        # task: survival analysis
        elif self.task == 3:
            raise NotImplementedError(
                "Task 3 not implemented for DFCI dataset yet")
            """ lsDownloadCaseSubmitterId = [s[:12] for s in lsDownloadFoldID]
            dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation.drop_duplicates(subset = 'case_submitter_id', ignore_index = True))
            dfDownload = pd.DataFrame({
                'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            dfClinicalInformation = pd.merge(dfClinicalInformation, dfDownload, on = "case_submitter_id")
            le = LabelEncoder()
            dfClinicalInformation.event =  le.fit_transform(dfClinicalInformation.event.values)
            leLabel = le.classes_
            self.dictInformation['event'] = leLabel

            scaler = MinMaxScaler()
            dfClinicalInformation['T'] = scaler.fit_transform(dfClinicalInformation['T'].values.reshape(-1, 1))

            stage_map = {
                'Stage I': 0,
                'Stage IA': 0,
                'Stage IB': 0,
                'Stage II': 1,
                'Stage IIA': 1,
                'Stage IIB': 1,
                'Stage IIC': 1
            }
            dfClinicalInformation['stage'] = dfClinicalInformation['stage'].map(stage_map) """
            
        elif self.task == 4:
            
            dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation)
            dfDownload = pd.DataFrame({
                # 'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            ##
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="folder_id", how='inner')

            dfClinicalInformation['label_old'] = dfClinicalInformation.label.copy()
            dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
                dfClinicalInformation.label.values)
            leLabel = le.classes_
            self.dictInformation["label"] = leLabel
        ############################################
        # custom tasks in string format.
        # in this case, the task name specifies the label column in the dataset.
        elif type(self.task) == str:

            dfClinicalInformation = self.fReduceDataFrame(dfClinicalInformation)
            dfDownload = pd.DataFrame({
                # 'case_submitter_id': lsDownloadCaseSubmitterId,
                'folder_id': lsDownloadFoldID
            })
            ##
            dfClinicalInformation = pd.merge(
                dfClinicalInformation, dfDownload, on="folder_id", how='inner')
            
            le = LabelEncoder()
            dfClinicalInformation.loc[:, 'label_old'] = dfClinicalInformation.label.copy(
            )
            dfClinicalInformation.loc[:, 'label'] = le.fit_transform(
                dfClinicalInformation.label.values)
            leLabel = le.classes_
            self.dictInformation['label'] = leLabel
        else:
            raise ValueError(f"Invalid task: {self.task}")

        dfClinicalInformation = self.fTransSensitive(
            dfClinicalInformation, exact=True).reset_index(drop=True)
        dfClinicalInformation.sensitive = le.fit_transform(
            dfClinicalInformation.sensitive.values)
        leSensitive = le.classes_
        self.dictInformation['sensitive'] = leSensitive

        if self.task != 2:
            # if not tumor detection, assume one label per patient and drop duplicates
            # then stratify by sensitive attribute and label
            dfDummy = dfClinicalInformation.drop_duplicates(
                subset='case_submitter_id', ignore_index=True).copy()
            if self.task == 3:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['event'])).tolist()
            else:
                dfDummy['fold'] = (
                    10*np.array(dfDummy['sensitive'].tolist())+np.array(dfDummy['label'])).tolist()
            dfDummy.fold = le.fit_transform(dfDummy.fold.values)
            foldNum = [0 for _ in range(int(len(dfDummy.index)))]
        else:
            raise NotImplementedError("We need a custom stratification for tumor detection task because slides from the same patient can have different labels.\n Sophie please help implement this.")

        if self.fold == 0:
            # no split. All data are testing data. For external validation
            bags = [[], [], list(range(len(dfDummy.index)))]

        elif self.fold == 1:
            train, valitest = train_test_split(
                dfDummy, train_size=0.6, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
            vali, test = train_test_split(
                valitest, test_size=0.5, random_state=self.seed, shuffle=True, stratify=valitest['fold'].tolist())
            # split training data
            if split < 1.0:
                train, remainder = train_test_split(
                    train, train_size=split, random_state=self.seed, shuffle=True, stratify=train['fold'].tolist())
                bags = [list(train.index), list(vali.index), list(test.index)]
                print("The length of training set: ", len(remainder))
            else:
                bags = [list(train.index), list(vali.index), list(test.index)]

        elif self.fold == 2:
            ab, cd = train_test_split(
                dfDummy, train_size=0.5, random_state=self.seed, shuffle=True, stratify=dfDummy['fold'].tolist())
            a, b = train_test_split(
                ab, test_size=0.5, random_state=self.seed, shuffle=True, stratify=ab['fold'].tolist())
            c, d = train_test_split(
                cd, train_size=0.5, random_state=self.seed, shuffle=True, stratify=cd['fold'].tolist())
            bags = [list(a.index), list(b.index), list(c.index), list(d.index)]

        for fid, indices in enumerate(bags):
            for idx in indices:
                foldNum[idx] = fid
        dfDummy['fold'] = foldNum

        if self.task == 3:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfRemoveDupDistribution = dfDummy.groupby(
                ['fold', 'label', 'sensitive']).size()

        dfDummy = dfDummy[['case_submitter_id', 'fold']]
        dfClinicalInformation = pd.merge(
            dfClinicalInformation, dfDummy, on="case_submitter_id")

        if isinstance(self.strEmbeddingPath, dict):
            # if embeddings span multiple folders
            df_pt = pd.DataFrame(
                {'folder_id': lsDownloadFoldID, 'path': lsDownloadPath})
            dfClinicalInformation = dfClinicalInformation.merge(
                df_pt, on='folder_id', how='inner')
        else:
            dfClinicalInformation['path'] = [
                f'{self.strEmbeddingPath}{p}.pt' for p in dfClinicalInformation['folder_id']]

        if self.task == 3:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'event', 'sensitive']).size()
        else:
            self.dfDistribution = dfClinicalInformation.groupby(
                ['fold', 'label', 'sensitive']).size()
        return dfClinicalInformation


class CancerDataset(Dataset):

    def __init__(self, df, task, fold_idx=None, feature_type='tile',
                 split_type="kfold",
                 exp_idx=0,
                 max_train_tiles=None,
                 ):
        '''
        df: dataframe containing the paths to the patches
        task: 1: cancer classification, 2: tumor detection, 3: survival analysis, 4: genetic classification
        fold_idx: 0: training, 1: validation, 2: testing
        split_type: kfold or vanilla
        exp_idx: current fold index (I know it's confusing)
        '''
        PARTITION_TYPE_MAP = {0: 'training', 1: 'validation', 2: 'testing', None: 'all'}
        self.partition_type = PARTITION_TYPE_MAP[fold_idx]
        self.feature_type=feature_type
        self.df = df
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.exp_idx = exp_idx
        self.task = task
        self.max_train_tiles = max_train_tiles
        self.sampler = None
        self.initialize_df()
        self.initialize_features()

    def initialize_df(self, df=None):
        '''
        Filter the dataframe based on the fold index and experiment index
        '''
        if df is not None:
            self.df = df
        ##
    
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
            # test ds generation
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
            return # no need to do anything for tile-level features
        elif self.feature_type == 'slide':
            # load all the features at once
            files = self.df['path'].unique()
            
            filenames = []
            features = []
            for file in files:
                with open(file, 'rb') as f:
                    feature = pickle.load(f)
                    filenames.extend(feature['filenames'])
                    features.append(feature['embeddings'])
            all_features = np.concatenate(features, axis=0).astype(np.float32)
            
            # fetch the features for each sample
            features = np.zeros((len(self.df), all_features.shape[1]))
            for idx, row in self.df.iterrows():
                filename = row['folder_id']
                idx_feature = filenames.index(filename)
                features[idx] = all_features[idx_feature]
            self.features = torch.from_numpy(features).float()
            

            
    def __getitem__(self, idx):
            

        row = self.df.loc[idx]
        # NOTE map_location
        if self.feature_type == 'slide':
            sample = self.features[idx]
        else:
            sample = torch.load(row.path, map_location='cpu')

        if self.max_train_tiles and self.partition_type == 'training':
            # randomly subsample the rows of the torch tensor
            if sample.shape[0] > self.max_train_tiles:
                idxs = np.random.choice(
                    sample.shape[0], self.max_train_tiles, replace=False)
                sample = sample[idxs, :]
        if self.task == 3:
            group = (row.sensitive)
            return sample, len(sample), row.sensitive, row.event, row['T'], group, row.stage, row.case_submitter_id, row.folder_id
        group = (row.sensitive, row.label)
        return sample, len(sample), row.sensitive, row.label, group, row.case_submitter_id, row.folder_id

    def __len__(self):
        return len(self.df)
    
    def get_groups(self):
        targets = set(self.df['label'].values)
        sensitives = set(self.df['sensitive'].values)
        if self.task == 3:
            return [s for s in sensitives]
        return [(s, t) for t in targets for s in sensitives]

class ReweightDataset(CancerDataset):
    def __init__(self, *args,
                 reweight_cols:List[Literal['sensitive', 'label']]=['sensitive', 'label'],
                 reweight_method: Literal['oversample',
                                          'undersample',
                                          'weightedsampler',
                                          'stablelearning'
                                          ] = 'undersample',
                 stablelearning_additional_args=['sensitive', 'label'],
                 **kwargs):
        '''
        reweight_cols: columns for 
        reweight_method:
            - oversample: oversample the minority group
            - undersample: undersample the majority group
            - weightedsampler: assign weights to the samples
        '''
        self.reweight_cols = reweight_cols
        self.reweight_method = reweight_method
        self.stablelearning_additional_args=stablelearning_additional_args
        
        super().__init__(*args, **kwargs)

    def initialize_df(self, df=None):
        # Select the dataframe based on the fold index and experiment index
        super().initialize_df(df)
        # perform reweighting
        # 1. estimate the number of samples in each group
        group_counts = self.df.groupby(self.reweight_cols).size()
        target_sample = group_counts.min(
        ) if self.reweight_method == 'undersample' else group_counts.max()
        # 2. resample the dataframe
        if self.reweight_method == 'oversample':
            self.df = self.df.groupby(self.reweight_cols).apply(
                lambda x: x.sample(target_sample, replace=True)).reset_index(drop=True)
        elif self.reweight_method == 'undersample':
            self.df = self.df.groupby(self.reweight_cols).apply(
                lambda x: x.sample(target_sample)).reset_index(drop=True)
        elif self.reweight_method == 'weightedsampler':
            ## estimate the weights for
            weight = 1 / (group_counts * len(group_counts))
            df_temp = self.df[self.reweight_cols].set_index(self.reweight_cols)
            sample_weights = weight.loc[df_temp.index].to_list()
            self.sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        elif self.reweight_method == 'stablelearning':
            if self.feature_type == 'tile':
                raise NotImplementedError("Stable learning is not implemented for tile-level features")
            self.initialize_features()
            # get the features for the additional columns
            X = self.features.numpy()
            if self.stablelearning_additional_args is not None:
                other_vars = []
                for col in self.stablelearning_additional_args:
                    if col in ['sensitive', 'label']:
                        # get one-hot encoding for the sensitive attribute
                        sensitive = self.df[col].values
                        sensitive = LabelEncoder().fit_transform(sensitive)
                        sensitive = OneHotEncoder().fit_transform(
                            sensitive.reshape(-1, 1)).toarray()
                        other_vars.append(sensitive)
                    else:
                        other_vars.append(self.df[col].values)
                X = np.concatenate([X] + other_vars, axis=1)
            sample_weights = decorrelation(X).flatten()
            self.sampler = WeightedRandomSampler(sample_weights, len(sample_weights))  
                
                
                        
            
            
            
        print(f"Dataset reweighted using {self.reweight_method} method")


def get_datasets(df, task, split_type, exp_idx, 
                 feature_type='tile',
                 reweight_method:Literal['oversample', 'undersample', 'weightedsampler', 'none']='none',
                 reweight_cols:List[Literal['sensitive', 'label']]=['sensitive', 'label'],
                 max_train_tiles=None):
    if split_type == 'kfold':
        if reweight_method != 'none':
            train_ds = ReweightDataset(
                df, task, 0, feature_type=feature_type,split_type=split_type, exp_idx=exp_idx, max_train_tiles=max_train_tiles,reweight_method=reweight_method,reweight_cols=reweight_cols)
        else:
            train_ds = CancerDataset(
                df, task, 0, feature_type=feature_type,split_type=split_type, exp_idx=exp_idx, max_train_tiles=max_train_tiles)

        val_ds = CancerDataset(
            df, task, 1, feature_type=feature_type,split_type=split_type, exp_idx=exp_idx)
        test_ds = CancerDataset(
            df, task, 2, feature_type=feature_type,split_type=split_type, exp_idx=exp_idx)
    elif split_type == 'vanilla':
        # train_ds = CancerDataset(df, task, 0, split_type=split_type,max_train_tiles=max_train_tiles)
        if reweight_method != 'none':
            train_ds = ReweightDataset(
                df, task, 0, feature_type=feature_type,split_type=split_type, max_train_tiles=max_train_tiles,reweight_method=reweight_method,reweight_cols=reweight_cols)
        else:
            train_ds = CancerDataset(
                df, task, 0, feature_type=feature_type,split_type=split_type, max_train_tiles=max_train_tiles)
        val_ds = CancerDataset(df, task, 1,  feature_type=feature_type,split_type=split_type)
        test_ds = CancerDataset(df, task, 2,  feature_type=feature_type,split_type=split_type)

    return train_ds, val_ds, test_ds

def survival_collate_fn(batch):
    # collate_fn for the dataloader (survival analysis)
    # training/validation: sample, length, sensitive, event, time, group, stage
    # testing: sample, length, sensitive, event, time, group, stage, case_submitter_id, folder_id
    if len(batch[0]) == 9:  # testing
        samples, lengths, sensitives, events, times, groups, stage, case_submitter_id, folder_id = zip(
            *batch)
    elif len(batch[0]) == 7:
        samples, lengths, sensitives, events, times, groups, stage = zip(
            *batch)
    else:
        raise ValueError(
            " invalid number of elements in the batch. Expected 7 or 9 elements in the batch for survival analysis.")
    # pad the samples
    max_len = max(lengths)
    padded_slides = []
    for i in range(0, len(samples)):
        pad = (0, 0, 0, max_len-lengths[i])
        padded_slide = torch.nn.functional.pad(samples[i], pad)
        padded_slides.append(padded_slide)
    padded_slides = torch.stack(padded_slides)
    if len(batch[0]) == 9:
        return padded_slides, lengths, torch.tensor(sensitives), torch.tensor(events), torch.tensor(times), groups, stage, case_submitter_id, folder_id
    return padded_slides, lengths, torch.tensor(sensitives), torch.tensor(events), torch.tensor(times), groups, stage


def collate_fn(batch):
    # collate_fn for the dataloader (all tasks other than survival analysis)
    # training/validation: sample, length, sensitive, label, group
    # testing: sample, length, sensitive, label, group, case_submitter_id, folder_id
    if len(batch[0]) == 7:
        samples, lengths, sensitives, labels, groups, case_submitter_id, folder_id = zip(
            *batch)
    elif len(batch[0]) == 5:
        samples, lengths, sensitives, labels, groups = zip(*batch)
    else:
        raise ValueError(
            " invalid number of elements in the batch. Expected 5 or 7 elements in the batch.")
    # pad the samples
    max_len = max(lengths)
    padded_slides = []
    for i in range(0, len(samples)):
        pad = (0, 0, 0, max_len-lengths[i])
        padded_slide = torch.nn.functional.pad(samples[i], pad)
        padded_slides.append(padded_slide)
    padded_slides = torch.stack(padded_slides)
    if len(batch[0]) == 7:
        return padded_slides, lengths, torch.tensor(sensitives), torch.tensor(labels), groups, list(case_submitter_id), list(folder_id)
    return padded_slides, lengths, torch.tensor(sensitives), torch.tensor(labels), groups

def slide_level_collate_fn(batch):
    # collate_fn for the dataloader (all tasks other than survival analysis)
    # training/validation: sample, length, sensitive, label, group
    # testing: sample, length, sensitive, label, group, case_submitter_id, folder_id
    if len(batch[0]) == 7:
        samples, lengths, sensitives, labels, groups, case_submitter_id, folder_id = zip(
            *batch)
    elif len(batch[0]) == 5:
        samples, lengths, sensitives, labels, groups = zip(*batch)
    else:
        raise ValueError(
            " invalid number of elements in the batch. Expected 5 or 7 elements in the batch.")
    samples = torch.stack(samples)
    if len(batch[0]) == 7:
        return samples, lengths, torch.tensor(sensitives), torch.tensor(labels), groups, list(case_submitter_id), list(folder_id)
    return samples, lengths, torch.tensor(sensitives), torch.tensor(labels), groups


class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, resample=False, group_nums=None):
        self.data_source = data_source
        self.group_indices = {group: [] for group in data_source.get_groups()}
        for i in range(len(data_source)):
            group = data_source[i][4]
            self.group_indices[group].append(i)
        self.total_size = sum(len(group)
                              for group in self.group_indices.values())
        self.batch_size = batch_size
        self.batch_num = self.total_size // self.batch_size
        self.resample = resample
        self.group_nums = group_nums

    def __iter__(self):
        batch_indices = []
        if self.resample == False:
            for i in range(self.batch_num):
                indices = []
                for group in self.group_indices:
                    indices.append(random.choice(self.group_indices[group]))
                while len(indices) < self.batch_size:
                    group = random.choice(list(self.group_indices.keys()))
                    indices.append(random.choice(self.group_indices[group]))
                batch_indices.append(indices)
        else:
            for i in range(self.batch_num):
                indices = []
                for group in self.group_nums:
                    indices += random.choices(
                        self.group_indices[group], k=self.group_nums[group])
                batch_indices.append(indices)
        return iter(batch_indices)

    def __len__(self):
        return self.total_size // self.batch_size

