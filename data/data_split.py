import pandas as pd
import os
from os import path
import numpy as np
import shutil

def complementary_list(total_list, sub_list):
    result_list = []
    for element in total_list:
        if not element in sub_list:
            result_list.append(element)
    return result_list

n_test = 0.2
csv_path = '../dataset/your_dataset'

results_path = '../list_by_diagnosis/your_dataset'
if not path.exists(results_path):
    os.makedirs(results_path)

train_path = os.path.join(results_path, 'train')#\list by diagnosis\train
if path.exists(train_path):
    shutil.rmtree(train_path)
os.makedirs(train_path)

test_path = os.path.join(results_path, 'test')#\list by diagnosis\test
if path.exists(test_path):
    shutil.rmtree(test_path)
os.makedirs(test_path)

diagnosis_df_paths = os.listdir(csv_path)
diagnosis_df_paths = [x for x in diagnosis_df_paths if x.startswith('0') or x.startswith('1')]

for diagnosis_df_path in diagnosis_df_paths:
    diagnosis_df = pd.read_csv(path.join(csv_path, diagnosis_df_path), sep=',', header=None,
                               names=['scan_dir', 'label'])

    diagnosis = diagnosis_df_path.split('.')[0]
    num_test = int(n_test * len(diagnosis_df))

    idx = np.arange(len(diagnosis_df))

    idx_test = np.random.choice(idx, size=num_test, replace=False)
    idx_test.sort()
    idx_train = complementary_list(idx, idx_test)

    test_df = diagnosis_df.loc[idx_test]
    print('Label', diagnosis, 'the number of test:', len(test_df))
    train_df = diagnosis_df.loc[idx_train]

    train_df.to_csv(path.join(train_path, str(diagnosis) + '.csv'), sep=',',
                    index=False, header=None)  # \list by diagnosis\cla\train\*.tsv
    test_df.to_csv(path.join(test_path, str(diagnosis) + '.csv'), sep=',', index=False, header=None)

n_val = 0.25 # 0.8*0.25 = 0.2
csv_path = r'../list_by_diagnosis/your_dataset_train/train'

results_path = r'../list_by_diagnosis/your_dataset_train/train'

train_path = os.path.join(results_path, 'train')#\list by diagnosis\train
if path.exists(train_path):
    shutil.rmtree(train_path)
os.makedirs(train_path)

test_path = os.path.join(results_path, 'validation')#\list by diagnosis\validation
if path.exists(test_path):
    shutil.rmtree(test_path)
os.makedirs(test_path)

diagnosis_df_paths = os.listdir(csv_path)
diagnosis_df_paths = [x for x in diagnosis_df_paths if x.startswith('0') or x.startswith('1')]

for diagnosis_df_path in diagnosis_df_paths:
    diagnosis_df = pd.read_csv(path.join(csv_path, diagnosis_df_path), sep=',', header=None,
                               names=['scan_dir', 'label'])

    diagnosis = diagnosis_df_path.split('.')[0]
    num_test = int(n_val * len(diagnosis_df))

    idx = np.arange(len(diagnosis_df))

    idx_test = np.random.choice(idx, size=num_test, replace=False)
    idx_test.sort()
    idx_train = complementary_list(idx, idx_test)

    test_df = diagnosis_df.loc[idx_test]
    print('Label', diagnosis, 'the number of validation:',len(test_df))
    train_df = diagnosis_df.loc[idx_train]
    print('Label', diagnosis, 'the number of train:', len(train_df))

    train_df.to_csv(path.join(train_path, str(diagnosis) + '.csv'), sep=',',
                    index=False, header=None)  # \list by diagnosis\cla\train\*.tsv
    test_df.to_csv(path.join(test_path, str(diagnosis) + '.csv'), sep=',', index=False, header=None)

###########################################################################################################
#             K-Fold
###########################################################################################################
from sklearn.model_selection import StratifiedKFold

n_splits=5
subset_name = 'validation'
results_path = r'../list_by_diagnosis/your_dataset_train/train'

train_path = path.join(results_path, 'train_splits-' + str(n_splits))#\lists_by_diagnosis\train\
if path.exists(train_path):
    shutil.rmtree(train_path)
os.makedirs(train_path)
for i in range(n_splits):
    os.mkdir(path.join(train_path, 'split-' + str(i)))

test_path = path.join(results_path, subset_name + '_splits-' + str(n_splits))#\lists_by_diagnosis\train\validation_splits-5
if path.exists(test_path):
    shutil.rmtree(test_path)
os.makedirs(test_path)
for i in range(n_splits):
    os.mkdir(path.join(test_path, 'split-' + str(i)))

diagnosis_df_paths = os.listdir(results_path)
diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.csv')]

for diagnosis_df_path in diagnosis_df_paths:
    diagnosis = diagnosis_df_path.split('.')[0]
    diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path), sep=',', header=None,
                               names=['scan_dir', 'ins_num', 'label'])

    diagnoses_list = list(diagnosis_df.label)

    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):
        train_index, test_index = indices

        test_df = diagnosis_df.iloc[test_index]
        train_df = diagnosis_df.iloc[train_index]
        train_df.to_csv(path.join(train_path, 'split-' + str(i), str(diagnosis) + '.csv'), sep=',', index=False, header=None)
        test_df.to_csv(path.join(test_path, 'split-' + str(i), str(diagnosis) + '.csv'), sep=',', index=False, header=None)