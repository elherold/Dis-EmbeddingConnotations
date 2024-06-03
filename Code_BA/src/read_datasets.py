import pandas as pd
import numpy as np
import json

def read_file_in_chunks(file_path, chunk_size=10000):
    """
    Reads a large file in chunks and yields each chunk as a DataFrame.
    
    Parameters:
    file_path (str): The path to the file to be read.
    chunk_size (int): The number of lines per chunk. Default is 10,000.
    
    Yields:
    DataFrame: A DataFrame containing a chunk of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        chunk = []
        for i, line in enumerate(file):
            chunk.append(json.loads(line.strip()))
            if (i + 1) % chunk_size == 0:
                yield pd.DataFrame(chunk)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk)

def process_chunk_comments(chunk):
    """
    Processes a chunk of data by keeping relevant columns, handling missing values,
    and removing rows with deleted or removed comments.
    
    Parameters:
    chunk (DataFrame): The chunk of data to process.
    
    Returns:
    DataFrame: The processed chunk of data.
    """
    expected_columns = ["created_utc", "name", "body", "author", "subreddit", "link_id", "parent_id"]
    available_columns = [col for col in expected_columns if col in chunk.columns]
    
    if not available_columns:
        return pd.DataFrame()  # Return an empty DataFrame if none of the expected columns are available

    chunk = chunk[available_columns].copy()  # Ensure we are working with a copy

    if 'body' in chunk.columns:
        missing_values_count = chunk['body'].isnull().sum()
        if missing_values_count != 0:
            print(f"Missing values count in chunk: {missing_values_count}")

        # Replace '[deleted]' and '[removed]' comments with None (np.nan) using .loc
        chunk.loc[chunk['body'].str.contains(r'\[deleted\]', na=False), 'body'] = np.nan
        chunk.loc[chunk['body'].str.contains(r'\[removed\]', na=False), 'body'] = np.nan

        # Remove rows with missing or deleted comments
        chunk = chunk.dropna(subset=['body'])
        chunk = chunk[chunk['body'] != '']  # Remove empty comments
        chunk = chunk.drop_duplicates(subset='body')
    else:
        print("No 'body' column found in the chunk.")
    
    # Filter for parent comments
    if 'link_id' in chunk.columns and 'parent_id' in chunk.columns:
        chunk = chunk[chunk['link_id'] == chunk['parent_id']]

    return chunk

def process_chunk_submissions(chunk):
    """
    Processes a chunk of data by keeping relevant columns, handling missing values,
    and removing rows with deleted or removed comments.
    
    Parameters:
    chunk (DataFrame): The chunk of data to process.
    
    Returns:
    DataFrame: The processed chunk of data.
    """

    expected_columns = ["created_utc", "title", "selftext", "author", "subreddit", "is_self", "id"]
    available_columns = [col for col in expected_columns if col in chunk.columns]
    
    if not available_columns:
        print("No valid columns found in the chunk.")
        return pd.DataFrame()  # Return an empty DataFrame if none of the expected columns are available

    chunk = chunk[available_columns].copy()  # Ensure we are working with a copy

    print(f"Initial chunk size: {chunk.shape}")

    # Replace '[deleted]' and '[removed]' comments with None (np.nan) using .loc
    chunk.loc[chunk['selftext'].str.contains(r'\[deleted\]', na=False), 'selftext'] = np.nan
    chunk.loc[chunk['selftext'].str.contains(r'\[removed\]', na=False), 'selftext'] = np.nan

    if 'selftext' in chunk.columns:

        # Remove rows with missing or deleted comments
        chunk = chunk.dropna(subset=['selftext'])
        #print(f"Chunk size after dropping empty comments: {chunk.shape}")
        chunk = chunk[chunk['selftext'] != '']  # Remove empty comments
        #print(f"Chunk size after dropping empty comments: {chunk.shape}")   
        chunk = chunk.drop_duplicates(subset='selftext')
        #print(f"Chunk size after dropping duplicates: {chunk.shape}")
    else:
        print("No 'selftext' column found in the chunk.")

    return chunk

def create_dataframe(file_path):
    """
    Reads a file in chunks, processes each chunk, and concatenates the processed chunks into a single DataFrame.
    
    Parameters:
    file_path (str): The path to the file to be read and processed.
    
    Returns:
    DataFrame: The final concatenated DataFrame after processing all chunks.
    """
    processed_chunks = []
    for chunk in read_file_in_chunks(file_path):
        if "comments" in file_path:
            #print("Processing comments...")
            processed_chunk = process_chunk_comments(chunk)
        elif "submissions" in file_path:
            #print("Processing submissions...")
            processed_chunk = process_chunk_submissions(chunk)
        if not processed_chunk.empty:
            processed_chunks.append(processed_chunk)
            print(f"Processed chunk size: {processed_chunk.shape}")

    if processed_chunks:
        df = pd.concat(processed_chunks, ignore_index=True)
        # call selftext column if it exists body for consistency
        if 'selftext' in df.columns:
            df.rename(columns={'selftext': 'body'}, inplace=True)
        print(f"Final DataFrame size: {df.shape}")

        return df
    else:
        df = pd.DataFrame()  # Handle the case where no chunks were processed
        print("No valid data was processed.")
        
        return None
"""
def main():
    file_path = "data/data_raw/antifeminists_submissions"
    file_path2 = "data/data_raw/AskFeminists_submissions"
    df = create_dataframe(file_path)
    print(df.head())

if __name__ == "__main__":
    main()
"""
    
