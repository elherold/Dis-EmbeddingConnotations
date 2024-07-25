import os
import csv

def process_csv_in_chunks(file_path, parent_dir, processed_folder, chunk_size=10000):
    """
    Process a large CSV file in chunks, removing URLs and splitting into lines of 10 words each.
    Saves the processed data incrementally to avoid high memory usage.

    Parameters:
    file_path (str): Path to the CSV file to process.
    parent_dir (str): The parent directory name to use as the key in processed_data dictionary.
    processed_folder (str): The folder where processed files will be saved.
    chunk_size (int): Number of rows to read per chunk.

    Returns:
    None
    """
    words = []
    output_file_path = os.path.join(processed_folder, f"{parent_dir}.csv")

    with open(file_path, 'r', encoding='utf-8') as csvfile, open(output_file_path, 'a', newline='', encoding='utf-8') as output_csvfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(output_csvfile)

        for row in reader:
            for field in row:
                if "http" not in field:  # Filter out URLs
                    words.extend(field.split())  # Split field by spaces to get individual words
            
            # Process words into lines of 10 words
            while len(words) >= 10:
                line = words[:10]
                writer.writerow(line)
                words = words[10:]  # Remove processed words

        # Process any remaining words
        if len(words) >= 10:
            for i in range(0, len(words), 10):
                line = words[i:i + 10]
                if len(line) == 10:  # Only write lines with exactly 10 words
                    writer.writerow(line)

def load_and_process_csv_files(root_folder):
    """
    Loads and processes CSV files from a specified root folder. It reads the content of each file,
    concatenates the words, and then groups them into lines of exactly 10 words each. The results 
    are saved incrementally to avoid high memory usage.

    Parameters:
    root_folder (str): The path to the root folder containing the raw CSV files.

    Returns:
    None
    """
    # Set a large but reasonable CSV field size limit
    csv.field_size_limit(10**6)

    # Define the processed folder
    processed_folder = 'data/data_processed'
    os.makedirs(processed_folder, exist_ok=True)

    # Walk through the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        print(f"Processing directory: {dirpath}")
        parent_dir = os.path.basename(os.path.dirname(dirpath))
        for file in filenames:
            if file.endswith('.csv'):
                print(f"Found file: {file}")
                file_path = os.path.join(dirpath, file)

                print(f"Processing file: {file_path} under category {parent_dir}")
                process_csv_in_chunks(file_path, parent_dir, processed_folder)

                print(f"Finished processing file: {file_path}")

def main():
    """
    Main function to load and process raw CSV files, then save the processed data.
    """
    # Define the root folder path
    root_folder = 'data/data_raw'
    print(f"this is the currently selected root folder: {root_folder}")

    # load and process the raw CSV files into the right format
    load_and_process_csv_files(root_folder)
    print(f"processing of raw data done")


if __name__ == "__main__":
    main()
