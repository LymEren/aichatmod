from transformers import pipeline

# Pytorch based Keyword extraction ai sample project 

# Load the pre-trained model for keyword extraction
model = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")


# Function to extract keywords from a given text
def extract_keywords(text):
    # Use the pre-trained model to extract named entities (keywords)
    keywords = model(text)

    # Extract only the entities that represent keywords
    keywords = [entity['word'] for entity in keywords if entity['entity'] != 'O']

    return keywords


# Example usage
text = "PyTorch is a popular deep learning framework. It provides powerful tools for building and training neural networks."
keywords = extract_keywords(text)

# Print the extracted keywords
print(keywords)
