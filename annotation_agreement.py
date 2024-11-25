import os
import json
import math
import argparse
import krippendorff
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split

def format_emotion(emotion_value):
    """
    Formats the emotion value by parsing the JSON string and extracting the 'choices' key.
    If parsing fails, returns the original value in a list.

    Args:
        emotion_value (str): The raw emotion value as a string.

    Returns:
        list: A list of emotions extracted from the input value.
    """
    try:
        return json.loads(emotion_value.replace('""', '"'))['choices']
    except:
        return [emotion_value]


def process_row(row):
    """
    Processes a row of the data to extract emotions and their intensities.

    Args:
        row (pd.Series): A row of the DataFrame containing emotion intensity columns.

    Returns:
        dict: A dictionary where keys are emotion names and values are their respective intensities.
    """
    result = {}
    for col in emotion_columns:
        if pd.notna(row[col]):
            result[col.split('-')[-1].capitalize()] = row[col].split(':')[0].strip()
    if 'Neutral' in format_emotion(row['emotion']):
        result['Neutral'] = ''
    return result


def preprocess_data(data, emotions):
    """
    Preprocesses the data to create binary DataFrames for each emotion, indicating whether the emotion is present.

    Args:
        data (pd.DataFrame): The input DataFrame containing text, annotator, and emotions_present columns.
        emotions (list): A list of all emotions present in the dataset.

    Returns:
        dict: A dictionary where keys are emotion names and values are binary DataFrames indicating emotion presence.
    """
    filtered_data = data[['text', 'annotator', 'emotions_present']]

    binary_dataframes = {}
    for emotion in emotions:
        binary_dataframes[emotion] = filtered_data[['text', 'annotator']].copy()

        binary_dataframes[emotion] = binary_dataframes[emotion].drop_duplicates(subset=['annotator', 'text'])
        binary_dataframes[emotion][emotion] = filtered_data.apply(
            lambda row: 1 if emotion in row['emotions_present'].keys() else 0, axis=1
        )
        binary_dataframes[emotion] = binary_dataframes[emotion].pivot(index='annotator', columns='text', values=emotion)

    return binary_dataframes

def get_all_keys(dicts):
    """
    Gets all unique keys from a list of dictionaries.

    Args:
        dicts (list): A list of dictionaries.

    Returns:
        list: A list of all unique keys from the input dictionaries.
    """
    return list([k for d in dicts for k in d.keys()])

def calculate_emotion_intensity(row, emotion):
    """
    Calculates the intensity value of a given emotion for a specific row.

    Args:
        row (pd.Series): A row of the DataFrame containing emotion intensity columns.
        emotion (str): The emotion to calculate intensity for.

    Returns:
        int: The intensity value (0 if not found, 1 for Low, 2 for Medium, 3 for High).
    """
    intensity_str = row.get(f'emotion-intensity-{emotion.lower()}', '')
    if pd.notna(intensity_str):
        if 'Low' in intensity_str:
            return 1
        elif 'Medium' in intensity_str:
            return 2
        elif 'High' in intensity_str:
            return 3
    return 0

def calculate_emotion_distribution(df, emotions, set_name):
    """
    Calculates and prints the emotion distribution for the given dataset.

    Args:
        df (pd.DataFrame): The dataset for which to calculate the distribution.
        emotions (list): A list of all emotions present in the dataset.
        set_name (str): The name of the dataset (e.g., 'train', 'dev', 'test').
    """
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = df[emotion].sum()
    emotion_counts['Neutral'] = df['Neutral'].sum()

    print(f'Emotion distribution for {set_name} set:')
    for emotion, count in emotion_counts.items():
        print(f'{emotion}: {count}')
    print('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process emotion annotations and calculate Krippendorff\'s Alpha')
    parser.add_argument('--input', type=str, help='Path to the input CSV file', default='first_step.csv')
    parser.add_argument('--output', type=str, help='Path to the output files',  default='output')
    parser.add_argument('--merge_annotators', type=str, help='Merge annotations from multiple annotators', default='')
    parser.add_argument('--lang', type=str, help='Language of the dataset', default='Chinese',)
    parser.add_argument('--batch', type=str, help='Batch number of the dataset', default='1',)
    parser.add_argument('--if_split', type=bool, help='If each csv file split into train/dev/test', default=True)
    
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
    
    emotion_columns = [col for col in df.columns if 'emotion-intensity-' in col]
    df['emotions_present'] = df.apply(process_row, axis=1)
    
    if merge_annotators != '':
        input_dict = json.loads(args.merge_annotators)
        for merges in input_dict.values():
            new_ann = '&'.join(merges)
            for merge in merges:
                df.loc[df.annotator == merge, 'annotator'] = new_ann
                
    df[['annotation_id', 'text', 'annotator', 'emotions_present']].to_csv(
        os.path.join(output_path, 'individual_annotations.csv'), index=False
    )
    emotions = [i for i in df.explode('emotions_present')['emotions_present'].unique().tolist() if pd.notna(i)]
    print(f'Emotion classes in dataset: {emotions}')
    annotations = preprocess_data(df, emotions)
    
    annotators = df.annotator.unique().tolist()
    print(f'\nAnnotators: {annotators}')
    print(f'Number of annotations by each annotator:')
    for annotator in annotators:
        print(f'{annotator}: {df[df.annotator == annotator].shape[0]}')
    print(f'\nNumber of annotators: {len(annotators)}')
    
    # process present emotions
    ann_df = pd.DataFrame(columns=['text'] + emotions)
    unique_texts = df.text.unique()
    ann_df['text'] = unique_texts
    
    # Dictionary to store average intensity for each emotion per text
    average_intensity_dict = {}  
    
    for text in unique_texts:
        text_df = df[df.text == text]
        # Get the number of unique annotators for this text
        annotator_count = text_df['annotator'].nunique()
        total_intensity = {emotion: 0 for emotion in emotions}
        for _, row in text_df.iterrows():
            for emotion in emotions:
                total_intensity[emotion] += calculate_emotion_intensity(row, emotion)
        
        #  if a text is neither joyful, fearful, angry, sad etc, we classify it as neutral.
        all_emotions_zero = True
        for emotion in emotions:
            average_intensity = total_intensity[emotion] / annotator_count
            if average_intensity > 0.5:
                ann_df.loc[ann_df.text == text, emotion] = 1
                all_emotions_zero = False
            else:
                ann_df.loc[ann_df.text == text, emotion] = 0
            average_intensity_dict[(text, emotion)] = average_intensity

        # If all emotions are zero, mark as Neutral
        if all_emotions_zero:
            ann_df.loc[ann_df.text == text, 'Neutral'] = 1
        else:
            ann_df.loc[ann_df.text == text, 'Neutral'] = 0

    ann_df.insert(0, 'text_id', value=ann_df.index)
    ann_df['text_id'] = ann_df['text_id'].apply(lambda x: f'{lang.lower()}_{batch}_{str(x + 1).zfill(5)}')
    ann_df.to_csv(os.path.join(output_path, f'processed_emotion_annotations.csv'), index=False)
    
    # Chinese has very rare fear annotations, so we need to split the dataset to ensure a balanced distribution
    # Split the dataset into train, dev, and test sets
    if args.if_split:
        emotion_counts = ann_df[emotions+ ['Neutral']].sum().sort_values()
        rarest_emotion = emotion_counts.idxmin()
        rarest_emotion_count = emotion_counts.min()
        print(f"Rarest emotion: {rarest_emotion} with {rarest_emotion_count} occurrences.")
        
        rarest_emotion_rows = ann_df[ann_df[rarest_emotion] > 0]
        remaining_rows = ann_df[~ann_df.index.isin(rarest_emotion_rows.index)]
        # Split at leat 5 rarest emotion rows to dev set
        rarest_dev, rarest_train = train_test_split(rarest_emotion_rows, test_size=len(rarest_emotion_rows) - 5, random_state=42)
        remaining_train, remaining_dev = train_test_split(remaining_rows, test_size=200 - len(rarest_dev), random_state=42)
        dev_set = pd.concat([rarest_dev, remaining_dev])
        train_set = pd.concat([rarest_train, remaining_train])
        # Split the train set into train and test sets
        train_set, test_set = train_test_split(train_set, test_size=0.5, random_state=42)
        
        train_set = train_set.sort_values(by='text_id')
        dev_set = dev_set.sort_values(by='text_id')
        test_set = test_set.sort_values(by='text_id')
        
        calculate_emotion_distribution(train_set, emotions, 'train')
        calculate_emotion_distribution(dev_set, emotions, 'dev')
        calculate_emotion_distribution(test_set, emotions, 'test')
        
        
        # Save the train, dev, and test sets to CSV files
        train_set.to_csv(os.path.join(output_path, 'train_emotion.csv'), index=False)
        dev_set.to_csv(os.path.join(output_path, 'dev_emotion.csv'), index=False)
        test_set.to_csv(os.path.join(output_path, 'test_emotion.csv'), index=False)

    
    
    # Calculate the intensity for each emotion and save to a new CSV file
    ann_intensity = ann_df.copy()
    for i, row in enumerate(ann_intensity.itertuples()):
        present_emotions = [emotion for emotion in emotions if getattr(row, emotion) == 1]
        if present_emotions:
            for emotion in present_emotions:
                average_intensity = average_intensity_dict[(row.text, emotion)]
                # Classify based on average score required by the guidelines
                intensity_val = min(3, round(average_intensity))  
                ann_intensity.loc[ann_intensity.text == row.text, emotion] = intensity_val

    ann_intensity.to_csv(os.path.join(output_path, f'processed_intensity_annotations.csv'), index=False)
    
    if args.if_split:
        train_intensity = ann_intensity[ann_intensity['text'].isin(train_set.text)]
        dev_intensity = ann_intensity[ann_intensity['text'].isin(dev_set.text)]
        test_intensity = ann_intensity[ann_intensity['text'].isin(test_set.text)]
        train_intensity.to_csv(os.path.join(output_path, 'train_intensity.csv'), index=False)
        dev_intensity.to_csv(os.path.join(output_path, 'dev_intensity.csv'), index=False)
        test_intensity.to_csv(os.path.join(output_path, 'test_intensity.csv'), index=False)
    
    
        

    
    # Statistical calculations for emotion counts
    emotion_counts = {}
    for emotion in emotions:
        emotion_counts[emotion] = annotations[emotion].sum().sum()
        
    
    with open(os.path.join(output_path, f'stats.txt'), 'w') as f:
        f.write(f'Number of annotations per emotion:\n')
        headers = ['Emotion', 'Count in annotations', 'Count after majority vote', "Krippendorff's Alpha"]
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
        
        # Calculate the number of texts with 0, 1, ... n emotions
        f.write('\n\nNumber of 0, 1, ... n emotions per text:\n')
        n_emotion_counts = {}
        for n in range(len(emotions) + 1):
            n_count = len(ann_df[ann_df[emotions].sum(axis=1) == n])
            if n_count > 0:
                n_emotion_counts[n] = n_count

        data_list = [[key, value] for key, value in n_emotion_counts.items()]
        headers = ['Number of Emotions', 'Count']
        f.write(tabulate(data_list, headers=headers, tablefmt='grid'))