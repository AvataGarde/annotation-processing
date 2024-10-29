# Copyright 2024 Idris Abdulmumin
#
# Adapted from https://github.com/emotion-analysis-project/emotion-dataset-pipeline/blob/main/scripts/agreement_potato.py
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script processes annotated emotion data and calculates Krippendorff's Alpha
# to measure the reliability of annotations from Label Studio annotated instances.

# To install dependencies, run:
# pip install pandas krippendorff

import os
import json
import math
import argparse
import krippendorff
import pandas as pd
from tabulate import tabulate

def format_emotion(emotion_value):
  try:
    return json.loads(emotion_value.replace('""', '"'))['choices']
  except:
    return [emotion_value]

def process_row(row):
  result = {}
  for col in emotion_columns:
    if pd.notna(row[col]):
      result[col.split('-')[-1].capitalize()] = row[col].split(':')[0].strip()
  if 'Neutral' in format_emotion(row['emotion']):
    result['Neutral'] = ''
  return result

def preprocess_data(data, emotions):
  filtered_data = data[['text', 'annotator', 'emotions_present']]

  binary_dataframes = {}
  for emotion in emotions:
    binary_dataframes[emotion] = filtered_data[['text', 'annotator']].copy()
    
    binary_dataframes[emotion] = binary_dataframes[emotion].drop_duplicates(subset=['annotator', 'text'])
    binary_dataframes[emotion][emotion] = filtered_data.apply(lambda row: 1 if emotion in row['emotions_present'].keys() else 0, axis=1)
    binary_dataframes[emotion] = binary_dataframes[emotion].pivot(index='annotator', columns='text', values=emotion)

  return binary_dataframes

def get_all_keys(dicts):
  return list([k for d in dicts for k in d.keys()])

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Process emotion annotations and calculate Krippendorff\'s Alpha')
  parser.add_argument('--input', type=str, help='Path to the input CSV file', required=True)
  parser.add_argument('--output', type=str, help='Path to the output files', default='')
  parser.add_argument('--merge_annotators', type=str, help='Merge annotations from multiple annotators', default='')
  parser.add_argument('--lang', type=str, help='Language of the dataset', required=True)
  parser.add_argument('--batch', type=str, help='Batch number of the dataset', required=True)

  args = parser.parse_args()

  input_path = args.input
  output_path = args.output
  lang = args.lang
  batch = args.batch
  merge_annotators = args.merge_annotators

  assert input_path.endswith('.csv') or input_path.endswith('.tsv'), 'Input file must be a CSV or TSV file'
  assert os.path.exists(input_path), f'Input file: {input_path} does not exist'

  if output_path != '':
    try:
      os.makedirs(output_path)
    except:
      pass
  else:
    print('No output path specified. Saving output files to the input file directory.')
    output_path = os.path.dirname(input_path)

  file_name = os.path.splitext(os.path.basename(input_path))[0]

  try:
    df = pd.read_csv(input_path)
  except:
    try:
      df = pd.read_csv(input_path, sep='\t')
    except:
      raise ValueError('Could not read input file. Please make sure the file is a valid CSV or TSV file.')

  emotion_columns = [col for col in df.columns if 'emotion-' in col]
  df['emotions_present'] = df.apply(process_row, axis=1)

  if merge_annotators != '':
    input_dict = json.loads(args.merge_annotators)
    for merges in input_dict.values():
      new_ann = '&'.join(merges)
      for merge in merges:
        df.loc[df.annotator == merge, 'annotator'] = new_ann

  df[['annotation_id', 'text', 'annotator', 'emotions_present']].to_csv(os.path.join(output_path, 'individual_annotations.csv'), index=False)
  # print(df.annotator.unique())

  emotions = [i for i in df.explode('emotions_present')['emotions_present'].unique().tolist() if pd.notna(i)]
  print(f'Emotion classes in dataset: {emotions}')

  annotations = preprocess_data(df, emotions)

  annotators = df.annotator.unique().tolist()
  majority_vote_value = len(annotators) // 2 + 1
  print(f'\nAnnotators: {annotators}')
  # print aumber of annotations by each annotator
  print(f'Number of annotations by each annotator:')
  for annotator in annotators:
    print(f'{annotator}: {df[df.annotator == annotator].shape[0]}')
  
  # determine if the sum of two annotators' annotations == len(annotations)

  print(f'\nNumber of annotators: {len(annotators)}')
  print(f'Majority vote value: {majority_vote_value}\n')

  # process present emotions
  ann_df = pd.DataFrame(columns=['text'] + emotions)
  unique_texts = df.text.unique()
  ann_df['text'] = unique_texts

  for text in unique_texts:
    text_df = df[df.text == text]
    emotions_present = text_df['emotions_present'].tolist()
    present_emotions = get_all_keys(emotions_present)

    for emotion in emotions:
      emotion_count = sum([1 for emotions in emotions_present if emotion in emotions])
      ann_df.loc[ann_df.text == text, emotion] = 1 if emotion_count >= majority_vote_value else 0

  ann_df.insert(0, 'text_id', value = ann_df.index)
  ann_df['text_id'] = ann_df['text_id'].apply(lambda x: f'{lang.lower()}_{batch}_{str(x + 1).zfill(5)}')

  ann_df.to_csv(os.path.join(output_path, f'processed_emotion_annotations.csv'), index=False)

  intensity_value = {
    'Low': 1,
    'Medium': 2,
    'High': 3
  }

  ann_intensity = ann_df.copy()
  for i, row in enumerate(ann_intensity.itertuples()):
    present_emotions = [emotion for emotion in emotions if row.__getattribute__(emotion) == 1]
    if present_emotions:
      temp_annotations = df[(df.text == row.text)].copy()
      r_emotions = temp_annotations['emotions_present'].to_list()
      for emotion in present_emotions:
        intensities = [r_emotion[emotion] if emotion in r_emotion else '0' for r_emotion in r_emotions]
        intensity_val = math.ceil(sum([intensity_value.get(i) if i in intensity_value else 0 for i in intensities]) / len(temp_annotations))

        ann_intensity.loc[ann_intensity.text == row.text, emotion] = intensity_val

  ann_intensity.to_csv(os.path.join(output_path, f'processed_intensity_annotations.csv'), index=False)

  # saving stats to file

  emotion_counts = {}
  for emotion in emotions:
    emotion_counts[emotion] = annotations[emotion].sum().sum()

  with open(os.path.join(output_path, f'stats.txt'), 'w') as f:
    f.write(f'Number of annotations per emotion:\n')
    headers = ['Emotion', 'Count in annotations', 'Count after majority vote', 'Krippendorff\'s Alpha']
    data_list = [[key, value] for key, value in emotion_counts.items()]
    for row in data_list:
      row.append(ann_df[ann_df[row[0]] == 1].shape[0])

    avg = 0
    emotion_alphas = {}
    for emotion in annotations:
      alpha = krippendorff.alpha(annotations[emotion].values)
      avg += alpha

      emotion_alphas[emotion] = alpha

    avg /= len(emotions)
    emotion_alphas['Average'] = avg

    for row in data_list:
      row.append(emotion_alphas[row[0]])

    data_list.append(['Total', sum(emotion_counts.values()), sum([row[2] for row in data_list]), avg])
      
    f.write(tabulate(data_list, headers=headers, tablefmt='grid'))

    f.write('\n\nNumber of 0, 1, ... n emotions per text:\n')
    n_emotion_counts = {}
    for n in range(len(emotions) + 1):
      n_count = len(ann_df[ann_df[emotions].sum(axis=1) == n])
      if n_count > 0:
        n_emotion_counts[n] = n_count

    data_list = [[key, value] for key, value in n_emotion_counts.items()]
    headers = ['Number of Emotions', 'Count']
    f.write(tabulate(data_list, headers=headers, tablefmt='grid'))