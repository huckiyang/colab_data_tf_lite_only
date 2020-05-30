import numpy as np
import pandas as pd
import pickle


def load_data_2020(feat_path, csv_path, feat_dim, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values

        feat_mtx = []
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.' + file_type
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels


LM_val, y_val = load_data_2020(feat_path, val_csv, 128, 'logmel')

print(LM_val[0:1100].shape, "--LM----")

#for n in range(0, 1000):
for n in range(7):
  k = 150*n
  LM_co = LM_val[0+k:150+k]
  np.save('LM_co_'+str(k), LM_co)
  print(y_val[0+k:150+k].shape, "--y----")

y_co = y_val[0:1100]
np.save('y_co', y_co)
print(y_val[0:3])
