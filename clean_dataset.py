"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                       FILE NAME: clean_dataset.py                         ║
║                                                                           ║
║  DESCRIPTION:                                                             ║
║  This file processes the dataset.csv file containing sentence, image urls ║
║  for positive (0) and negative (1) classification. It checks the          ║
║  accessibility of each image, removes entries with inaccessible images,   ║
║  and saves the cleaned dataset to a new CSV file.                         ║
║                                                                           ║
║  OUTPUT:                                                                  ║
║  - The script will output 'clean_dataset.csv' with only accessible images.║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# Imports
import requests  
from PIL import Image 
import io  
import pandas as pd  
import os  

# Function to download an image
def download_image(index, url, indexToRemove):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        img = Image.open(io.BytesIO(response.content))
        # img.save(save_path)
        # print(f"Downloaded and saved {url} to {save_path}")
    except requests.exceptions.Timeout:
        print(f"Index: {index} || Timeout: {url} took too long to download.")
        indexToRemove.append(index)
    except Exception as e:
        print(f"Index: {index} || Failed to download : {e}" )
        indexToRemove.append(index)

def main():
    file_path = os.path.join(os.getcwd(), "dataset.csv")
    df = pd.read_csv(file_path)

    # Extract the sentence
    sentence = df['sentence']
    image_pos = df['pos_url']
    image_neg = df['neg_url']

    df = pd.DataFrame({
        'sentence': sentence,
        'image_1': image_pos,
        'image_2': image_neg})

    df['label_image1'] = 0
    df['label_image2'] = 1

    print("Note ->  0: positive, 1: negative")

    # Check images that can be downloaded
    pos_indexToRemove = []
    neg_indexToRemove = []

    # Iterate over the URLs in the dataset
    print("\nImage Positive")
    for index, row in df.iterrows():
        url_pos = row['image_1']  # Replace 'image_url' with the column name in your CSV
        download_image(index, url_pos, pos_indexToRemove)
    print(f"Indicies to remove for positive images: {pos_indexToRemove}")

    print("\nImage Negative ")
    for index, row in df.iterrows():
        url_neg = row['image_2']  # Replace 'image_url' with the column name in your CSV
        download_image(index, url_neg, neg_indexToRemove)
    print(f"Indicies for negative: {neg_indexToRemove}")

    # merge the two indicies above
    merged_toRemove = list(set(neg_indexToRemove) | set(pos_indexToRemove))
    merged_toRemove.sort()

    for index, row in df.iterrows():
        if index in merged_toRemove:
            # print("Removing index: ", index)
            df = df.drop(index = index)

    # Save the DataFrame to a CSV file
    file_path = 'clean_dataset.csv'
    df.to_csv(file_path, index=False)
    print("File saved as clean_dataset.csv")


main()
