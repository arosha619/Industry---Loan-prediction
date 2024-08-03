from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("dilkasithari-IT/news_data")
all_industries = list(set(dataset['train']['industry']))

pipe = pipeline(
                "text-classification", 
                model="dilkasithari-IT/fine-tuned-twitter-roberta-base-sentiment-latest",
                device=0
                )

def get_industries(specific_industry):
    filtered_dataset = dataset.filter(lambda example: example['industry'] == specific_industry)
    filtered_dataset = filtered_dataset.rename_column('combined_text', 'text')
    return filtered_dataset

def inference_news(specific_industry):
    assert specific_industry in all_industries, f"Industry {specific_industry} not found in the dataset"
    filtered_dataset = get_industries(specific_industry)
    result = pipe(
                inputs=filtered_dataset['train']['text'],
                batch_size=50
                )

    # Separate the scores by label
    positive_scores = [item['score'] for item in result if item['label'] == 'positive']
    negative_scores = [item['score'] for item in result if item['label'] == 'negative']

    # Calculate the average score for each label
    positive_score_percentage = len(positive_scores) / len(result)
    negative_score_percentage = len(negative_scores) / len(result)

    print(f"Percentage of Positive Score: {positive_score_percentage}")
    print(f"Percentage of Negative Score: {negative_score_percentage}")

inference_news('Utilities')