#sort lesion types into train/test/validate
import pandas as pd
import os
import shutil
import random
import csv

# Set the seed for reproducibility
seed = 1
random.seed(seed)

# Define directories
data_dir = r'C:/Users/grayj/Desktop/DATASET/BothImages'
train_dir = r'C:/Users/grayj/Desktop/DATASET/Train'
test_dir = r'C:/Users/grayj/Desktop/DATASET/Test'
validation_dir = r'C:/Users/grayj/Desktop/DATASET/Validate'

# Create subdirectories for train, test, and validation if they don't exist
for subdir in ['benign', 'malignant']:
    os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, subdir), exist_ok=True)

# Initialize counters for statistics
train_examples = test_examples = validation_examples = 0

# Define the output CSV files
train_csv = r'C:/Users/grayj/Desktop/DATASET/train.csv'
test_csv = r'C:/Users/grayj/Desktop/DATASET/test.csv'
validation_csv = r'C:/Users/grayj/Desktop/DATASET/valid.csv'

# Create CSV writers for each set
with open(train_csv, 'w', newline='') as train_file, \
     open(test_csv, 'w', newline='') as test_file, \
     open(validation_csv, 'w', newline='') as validation_file:
    
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)
    validation_writer = csv.writer(validation_file)
    
    # Write headers to each CSV
    train_writer.writerow(['image_id', 'dx'])
    test_writer.writerow(['image_id', 'dx'])
    validation_writer.writerow(['image_id', 'dx'])

    # Open and read the CSV file containing metadata
    metadata_file = r'C:/Users/grayj/Desktop/DATASET/HAM10000_metadata.csv'
    skin_df2 = pd.read_csv(metadata_file)

    # Loop over each row in the CSV and decide whether to assign it to train/test/validation
    for index, row in skin_df2.iterrows():
        img_file = row['image_id']
        label = row['dx']

        # Random number to decide whether it's train/test/validation
        random_num = random.random()

        # Determine the destination folder and CSV writer
        if random_num < 0.8:  # 80% training
            location = train_dir
            csv_writer = train_writer
            train_examples += 1
        elif random_num < 0.9:  # 10% validation
            location = validation_dir
            csv_writer = validation_writer
            validation_examples += 1
        else:  # 10% test
            location = test_dir
            csv_writer = test_writer
            test_examples += 1

        # Set label to benign or malignant
        if label in ['nv', 'bkl', 'df']:  # Benign categories
            classification = 'benign'
        elif label in ['mel', 'bcc', 'akiec', 'vas']:  # Malignant categories
            classification = 'malignant'
        else:
            continue  # Skip if label is not recognized

        # Construct source and destination paths
        source_path = os.path.join(data_dir, img_file + '.jpg')
        dest_path = os.path.join(location, classification, img_file + '.jpg')

        # Copy image to the corresponding directory if it exists and write to the CSV
        if os.path.exists(source_path):
            try:
                shutil.copy(source_path, dest_path)
                print(f"Copied {source_path} to {dest_path}")

                # Write image_id and dx to the corresponding CSV file
                csv_writer.writerow([img_file, label])
                
            except Exception as e:
                print(f"Error copying {source_path} to {dest_path}: {e}")
        else:
            print(f"Source file not found: {source_path}")

# Print statistics
print(f'Number of training examples: {train_examples}')
print(f'Number of validation examples: {validation_examples}')
print(f'Number of test examples: {test_examples}')