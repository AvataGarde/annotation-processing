# Annotation Processing  

This script processes annotated emotion data from Label Studio and calculates Krippendorff's Alpha to measure the reliability of annotations. It outputs processed files with individual and majority-vote emotion annotations, intensity values, and summary statistics.

```bash
pip install pandas krippendorff tabulate

```

### Usage

To run the script, use the following command in the terminal or run script in  `annotation_agreement.ipynb` Notebook:

```
python annotation_agreement.py \
  --input <path_to_input_file> \
  --output <path_to_output_directory> \
  --lang <language> \
  --batch <batch_number>
```


### Parameters

`--input:` Path to the input CSV file with raw emotion annotations.

`--output:` Directory where the processed files will be saved.

`--lang:` Language of the dataset. (Example: "Hausa")

`--batch:` Batch number of the dataset. This helps identify data for specific batches.

`--merge_annotators (optional):` A JSON string to merge annotations from multiple annotators. Use this option if you want to combine annotations from specific annotators. Example: ```json '{"merge1": ["meryembechar4@gmail.com", "khaoulafarouz@gmail.com"]}'```



### Output Files

The script generates several output files in the specified output directory:

`individual_annotations.csv:` Contains individual annotations processed per annotator.
`processed_emotion_annotations.csv:` Includes annotations after applying the majority vote.
`processed_intensity_annotations.csv:` Contains emotion intensity values (Low, Medium, High) for each text.
`stats.txt:` Summarizes the statistics:

- Count of annotations per emotion
- Count after majority vote
- Krippendorff's Alpha for inter-annotator reliability
- Distribution of the number of emotions per text
