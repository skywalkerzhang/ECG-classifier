from helper_code import *
from team_code import get_features
import numpy as np, os, sys, joblib
import requests
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
import heartpy as hp

data_directory = "../data/WFDB_StPetersburg"
dataset_name = "WFDB_StPetersburg"

dataset_names = ['WFDB_CPSC2018', 'WFDB_CPSC2018_2', 'WFDB_Ga', 'WFDB_PTB', 'WFDB_PTBXL', 'WFDB_StPetersburg']

dataset_classes_fre = []
classes = []
scored = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC',
          'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB']
scored_classes_id = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003', '164909002', '251146004', '698252002', '10370003', '284470004', '427172004',  '164947007', '111975006', '164917005', '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '63593006', '164934002', '59931005', '17338001']
fre = [500, 500, 500, 1000, 500, 257]
times = []
'''
classes_abbr = pd.read_csv("../data/Dx_map.csv")


def get_classes_id(x):
    if x.Abbreviation in scored:
        return str(x['SNOMED CT Code'])


scored_classes_id = classes_abbr.apply(get_classes_id, axis=1)
scored_classes_id = scored_classes_id[scored_classes_id.notna()]
scored_classes_id = list(scored_classes_id)

'''

'''
for dataset_name in dataset_names:
    data_directory = "../data/" + dataset_name
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    # Find header and recording files.

    if not num_recordings:
        raise Exception('No data was provided.')

    tmp = set()
    # Extract classes from dataset.
    for header_file in header_files:
        header = load_header(header_file)
        tmp |= set(get_labels(header))

    for x in tmp:
        classes.append(x)

classes = np.unique(classes)

if all(is_integer(x) for x in classes):
    classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
else:
    classes = sorted(classes)  # Sort classes alphanumerically otherwise.
num_classes = len(classes)
'''
idx_fre = 0
for dataset_name in dataset_names:
    data_directory = "../data/" + dataset_name
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    # Extract features and labels from dataset.

    # print(classes)
    # ['10370003', '11157007', '17338001', '27885002', '29320008', ...]

    data = np.zeros((num_recordings, 14),
                    dtype=np.float32)  # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, len(scored_classes_id)), dtype=np.bool)  # One-hot encoding of classes
    time = np.zeros(num_recordings, dtype=np.float32)
    for i in range(num_recordings):
        # print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        # print(recording)
        # get times
        time[i] = float(get_num_samples(header)) / float(get_frequency(header))

        # Get age, sex and root mean square of the leads.
        age, sex, rms = get_features(header, recording, twelve_leads)
        data[i, 0:12] = rms
        data[i, 12] = age
        data[i, 13] = sex

        current_labels = get_labels(header)
        for label in current_labels:
            if label in scored_classes_id:
                j = scored_classes_id.index(label)
                labels[i, j] = 1

    # get the frequency of each dataset (each cols of labels)
    classes_fre = labels.sum(axis=0)
    times.append(time)

    # visualise classes
    fig = plt.figure()
    sns.set(font_scale=0.7)
    sns.barplot(scored, classes_fre)
    plt.title(dataset_name)
    pl.xticks(rotation=45)
    plt.savefig(fname="./vis/" + dataset_name + ".png", dpi=300)
    # plt.show()
    dataset_classes_fre.append(list(classes_fre))

    # visualization recording

    recording_vis = load_recording(recording_files[0])
    plt.figure(figsize=(20, 43))
    for i in range(0, 12):
        plt.subplot(12, 1, i + 1)
        plt.plot(np.arange(0, fre[idx_fre]) / fre[idx_fre], recording_vis[i, : fre[idx_fre]])
        plt.title(twelve_leads[i])
        plt.xticks(np.arange(0, 1, 0.04))
    plt.savefig(fname="./vis/ALl_" + dataset_name + ".png", dpi=300)
        # plt.show()
    idx_fre += 1

# draw a heatmap
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(50, 12))
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
sns.heatmap(dataset_classes_fre, linewidths=0.05, cmap=cmap, annot=True, fmt='g')
ax.set_yticklabels(dataset_names, rotation='horizontal')
ax.set_xticklabels(scored, rotation=45)
plt.savefig(fname="./vis/heatmap.png", dpi=250)
# plt.show()

# get mean time
mean_times = [np.mean(x) for x in times]
'''

label_vis = []
for i in range(len(classes)):
    if classes_fre[i] >= 10:
        label_vis.append([classes[i], classes_fre[i]])
        print("|%s|%d|" % (classes[i], classes_fre[i]))
# get the number of different sex
sex_fre = np.nansum(data[:, 13])
print("sex empty:", np.isnan(data[:, 13]).sum())
print(sex_fre)
print("ratio:", sex_fre / num_recordings)

# dealing with age
# center position
from copy import deepcopy

ages = data[:, 12]
print("age empty:", np.isnan(ages).sum())

from scipy.stats import mode

age_mean = np.nanmean(ages)
age_med = np.nanmedian(ages)
age_mode = mode(ages)
print(age_mean, age_med, age_mode)

# divergence
# 极差
age_max = np.nanmax(ages)
age_min = np.nanmin(ages)
age_ptp = age_max - age_min
# 方差
age_var = np.nanvar(ages)
# 标准差
age_std = np.nanstd(ages)
# 变异系数 Coefficient of Variation
age_coe = age_mean / age_std
print(age_max, age_min, age_ptp, age_var, age_std, age_coe)






# get names of classes

# def get_classes_name(q):
#     res = requests.get(
#         'https://data.bioontology.org/search?callback=jQuery224048026047835205743_1616512702132&q=%s&include_properties=false&include_views=false&obsolete=false&include_non_production=false&require_definition=false&exact_match=false&categories=&ontologies=&pagesize=150&apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&userapikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&format=jsonp&ncbo_slice=&_=1616512702133' % (
#             q))
#
#     rest = res.text
#
#     m = re.search('prefLabel":"(.+?)"', rest)
#     return m.group(1)
#
# classes_name = []
# for x in classes:
#     classes_name.append([x, get_classes_name(x)])
# output = pd.DataFrame(classes_name)
# output.to_csv(dataset_name + '_' + "mapping.csv", encoding="utf-8")
'''
