
#######################################################

#                 QUESTIONS TO ASK 
# (1) Check on the threshold counter to increase the word count 
#       Base frequent words is 45
#       Last frequent words is 35 (enough?)
# (2) Bias the filtering process to only pos_triplet or both pos_triplet and neg_triplet
#       pos_triplet is a reflection of the sentence 
#       both pos_triplet and neg_triplet would have a balanced for True/False


#                 NOTE:
# There is an balance to be found between the count_threshold and the vocabulary for the sentence
#  8 Threshold -> 33 words -> 220 sentences 
#  10 Threshold -> 27 words -> 190 sentences 

#  increase Threshold -> decrease word count -> decrease sentences

#######################################################

# Imports
import pandas as pd
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Functions 
def tokenize(text):
    text = text.lower()
    words = text.split(',')
    return words

def frequncy_counter(dataframe, column):
    word_counter = Counter()
    for text in dataframe[column]: 
        words = tokenize(text)
        word_counter.update(words)
    # Display the word counts
    # print("Word Frequencies:")
    # for word, count in word_counter.most_common():
    #     print(f"{word}: {count}")
    return word_counter


def filter(dataframe, threshold):

    counter_pos = frequncy_counter(dataframe, "pos_triplet")
    # counter_neg = frequncy_counter(dataframe, "neg_triplet")

    words_to_remove_pos = []
    words_to_remove_neg = []

    for word, count in counter_pos.items():
        # print(f"word = {word}, counter = {count}")
        if count < threshold:
            words_to_remove_pos.append(word)

    # for word, count in counter_neg.items():
    #     # print(f"word = {word}, counter = {count}")
    #     if count < threshold:
    #         words_to_remove_neg.append(word)

    index_to_remove = []

    for index, words in dataframe['pos_triplet'].items():
        # print(f"index = {index}, words = {words}") 
        word = tokenize(words)
        for word in tokenize(words):
            if word in words_to_remove_pos:
                # print(f" Remove this word = {word}, sentence = {words}")
                index_to_remove.append(index)

    # for index, words in dataframe['neg_triplet'].items():
    #     # print(f"index = {index}, words = {words}") 
    #     word = tokenize(words)
    #     for word in tokenize(words):
    #         if word in words_to_remove_neg:
    #             # print(f" Remove this word = {word}, sentence = {words}")
    #             index_to_remove.append(index)

    return(index_to_remove)

def display_batch(dataframe, counter):
    temp_pos = frequncy_counter(dataframe, "pos_triplet")
    # temp_neg = frequncy_counter(dataframe, "neg_triplet")

    print(f"* Batch {counter} of sample sentences * ")
    print("=======================")
    print(f"Number of sentenced:  {dataframe.shape[0]}")
    print(f" {len(temp_pos)} frequent words:")
    print(f" Positive word count: \n {temp_pos}\n")

    # Convert the Counter object to a dictionary
    word_freq = dict(temp_pos)  # Get the top 10 most common words

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(word_freq.keys(), word_freq.values())
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.title('Distribution')
    plt.xticks(rotation=45)
    plt.show()

    # print(f" Negative word count: \n {temp_neg}")
    print("\n")


def main():
    # =================
    # READ THE DATABASE
    # =================

    # Relative path to the file
    file_name = 'svo_probes.csv'
    file_path = os.path.abspath(file_name)
    df = pd.read_csv(file_path)

    print("* Databse stored as a dataframe * ")
    print("==================================")
    print(f"{df.head()}")
    print("\n")

    # ====================
    # DELETE MISSING ROWS
    # ====================

    # Filter rows containing '[]' in any column and delete
    rows_to_delete = df.map(lambda x: '[]' in str(x)).any(axis=1)
    df = df[~rows_to_delete]

    print("* Deleted empty rows * ")
    print("=======================")
    print(f"Number of rows to be deleted: {rows_to_delete.sum()}")
    print("\n")


    # =====================================================================================
    #                               GET THE 200 SENTENCES
    # ====================================================================================

    # ============================
    # FIND THE MOST FREQUENT WORDS
    # ============================

    word_counter_1 = frequncy_counter(df, "pos_triplet")
    # word_counter_1b = frequncy_counter(df, "neg_triplet")
    # word_counter_1 = word_counter_1a + word_counter_1b

    threshold = 400   # threshold for repeating vocabulary
    frequent_words = {word for word, count in word_counter_1.items() if count >= threshold}     # Identify frequent words >= threshold


    print("* Frequent words * ")
    print("=======================")
    print(f" {len(frequent_words)} frequent words:\n {frequent_words}")
    print("\n")


    # ========================================================
    # REMOVE WORD THAT OCCUR LESS THAN 3 TIMES IN ONE SENTENCE 
    # ========================================================
    rows_to_remove = []
    df_frequent_1 = df.copy()
    count_threshold = 3

    for index, text in df_frequent_1['pos_triplet'].items():
        count = 0 
        words = tokenize(text)
        for word in words:
            if word in frequent_words:
                count += 1 
        if count < count_threshold:
            # print("This row will be removed")
            rows_to_remove.append(index)


    df_frequent_2 = df_frequent_1.copy()   
    df_frequent_2 = df_frequent_2.drop(rows_to_remove) # dropping the words causes problems with the indexing

    # ================================
    # PICK 10 SENTENCES FROM EACH WORD
    # ================================

    threshold = 10
    rows_to_add = []    # rows to add to the new dataframe
    word_count = 0

    for freq_word in frequent_words:
        word_count += 1
        counter = 0 
        for index, word in df_frequent_2['pos_triplet'].items():
            if (freq_word in word) and (index not in rows_to_add):
                counter += 1
                # print(f" Counter : {counter}, Word found: {freq_word}, sentence: {df.iloc[index]['sentence']}")
                rows_to_add.append(index)

            if counter == threshold:
                # print(f" Counter reached 10, moving onto the next word")
                break

    # ==================================================
    # PICK REASONABLE BATCH SENTENCES FROM THE WORDS LISTED ABOVE
    # ==================================================
    df = df_frequent_1.iloc[rows_to_add].copy()

    count = 0 
    min_threshold = 4
    num_sentences = df.shape[0]
    while num_sentences > 200 and count < 10:
        index_to_remove = filter(df, min_threshold)
        df = df.drop(index_to_remove)
        # display_batch(df, count)
        num_sentences = df.shape[0]
        count += 1

        if count == 10:
            print("Manually Terminated")
            break

    display_batch(df, count)

    file_path = os.path.join(os.getcwd(), "database_v2.csv")
    df.to_csv(file_path, index=False)

    print("* Saving New Databse * ")
    print("=======================")
    print("file saved")




main()



