import numpy as np  # For handling arrays and calculating means
import pandas as pd

import nltk  # Natural Language Toolkit for BLEU score
nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu  # For BLEU score

from rouge_score import rouge_scorer  # For ROUGE score calculation

# If you're using a tokenizer (such as from nltk or other libraries), you may need to import it as well.
# Example (if using NLTK for tokenization):
from nltk.tokenize import word_tokenize


def CalculateAvgBLEUScore(Transcribed_Sentences, Validated_Sentences, tokenizer):
    BLEU_Scores = []
    for i in range(len(Transcribed_Sentences)):
        Valid_Sentence = tokenizer(Validated_Sentences[i])  # Call the function directly
        Transcribed_Sentence = tokenizer(Transcribed_Sentences[i])  # Access the string directly
        BLEU_Scores.append(nltk.translate.bleu_score.sentence_bleu([Valid_Sentence], Transcribed_Sentence))
    return np.mean(BLEU_Scores)

def CalculateAvgROUGEScore(Transcribed_Sentences, Validated_Sentences):
    rs = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ROUGE1_Percision_Scores = []
    ROUGE1_Recall_Scores = []
    ROUGE1_F1_Scores = []
    ROUGE2_Percision_Scores = []
    ROUGE2_Recall_Scores = []
    ROUGE2_F1_Scores = []
    ROUGEL_Percision_Scores = []
    ROUGEL_Recall_Scores = []
    ROUGEL_F1_Scores = []
    for i in range(len(Transcribed_Sentences)):
        rouge_score = rs.score(Validated_Sentences[i], Transcribed_Sentences[i])  # Access the string directly
        ROUGE1_Percision_Scores.append(rouge_score['rouge1'][0])
        ROUGE1_Recall_Scores.append(rouge_score['rouge1'][1])
        ROUGE1_F1_Scores.append(rouge_score['rouge1'][2])
        ROUGE2_Percision_Scores.append(rouge_score['rouge2'][0])
        ROUGE2_Recall_Scores.append(rouge_score['rouge2'][1])
        ROUGE2_F1_Scores.append(rouge_score['rouge2'][2])
        ROUGEL_Percision_Scores.append(rouge_score['rougeL'][0])
        ROUGEL_Recall_Scores.append(rouge_score['rougeL'][1])
        ROUGEL_F1_Scores.append(rouge_score['rougeL'][2])
    return {
        "rouge1": [np.mean(ROUGE1_Percision_Scores), np.mean(ROUGE1_Recall_Scores), np.mean(ROUGE1_F1_Scores)],
        "rouge2": [np.mean(ROUGE2_Percision_Scores), np.mean(ROUGE2_Recall_Scores), np.mean(ROUGE2_F1_Scores)],
        "rougeL": [np.mean(ROUGEL_Percision_Scores), np.mean(ROUGEL_Recall_Scores), np.mean(ROUGEL_F1_Scores)]
    }


# Assuming your CSV files have columns 'English' and 'Arabic'
def load_csv(file_path):
    return pd.read_csv(file_path)




# Prepare the data: tokenization
def prepare_data(csv_data):
    english_sentences = csv_data['English'].tolist()
    arabic_sentences = csv_data['Arabic'].tolist()  # This is the translation we will evaluate
    return english_sentences, arabic_sentences




# Example of how to load and evaluate both translations:
def evaluate_translations(csv_file_1, csv_file_2):
    # Load both CSV files
    csv_1 = load_csv(csv_file_1)
    csv_2 = load_csv(csv_file_2)

    # Prepare data for each CSV
    _, arabic_1 = prepare_data(csv_1)
    _, arabic_2 = prepare_data(csv_2)

    # Calculate BLEU scores
    tokenizer = word_tokenize  # NLTK tokenizer, you can change it if you're using another one
    bleu_1 = CalculateAvgBLEUScore(arabic_1, arabic_1, tokenizer)  # Compare against itself for consistency
    bleu_2 = CalculateAvgBLEUScore(arabic_2, arabic_1, tokenizer)

    # Calculate ROUGE scores
    rouge_1 = CalculateAvgROUGEScore(arabic_1, arabic_1)  # Compare against itself for consistency
    rouge_2 = CalculateAvgROUGEScore(arabic_2, arabic_1)

    return bleu_1, bleu_2, rouge_1, rouge_2



# Call the evaluation function with your CSV files
csv_file_1 = 'Translated_data_A.csv'  # using the translate library
csv_file_2 = 'Translated_data_R.csv'   #using chat gpt

bleu_1, bleu_2, rouge_1, rouge_2 = evaluate_translations(csv_file_1, csv_file_2)

# Output results
print(f"BLEU score for Translation 1: {bleu_1}")
print(f"BLEU score for Translation 2: {bleu_2}")
print(f"ROUGE score for Translation 1: {rouge_1}")
print(f"ROUGE score for Translation 2: {rouge_2}")
