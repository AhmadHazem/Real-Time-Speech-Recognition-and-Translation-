import pickle
import openai
import csv

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your OpenAI API key

# Step 1: Load English sentences from the .pkl file
with open('Transcribed_Sentences.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract English sentences from the loaded data
english_sentences = [item['text'] for item in data]

# Step 2: Translate each sentence using OpenAI's API

arabic_sentences = []
for sentence in english_sentences:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5 model instead of GPT-4
        messages=[
            {"role": "system", "content": "Translate the following sentence to Arabic:"},
            {"role": "user", "content": sentence}
        ]
    )
    # Extract the translated text
    translation = response['choices'][0]['message']['content'].strip()
    arabic_sentences.append(translation)

# Step 3: Create a CSV file with the English and Arabic sentences
with open('translation.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['English', 'Arabic'])  # Write header

    # Write each English-Arabic pair to the CSV file
    for english, arabic in zip(english_sentences, arabic_sentences):
        writer.writerow([english, arabic])

print("CSV file 'translation.csv' created successfully.")
