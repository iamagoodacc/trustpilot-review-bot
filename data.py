import requests
from bs4 import BeautifulSoup
import json
import uuid
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import spacy
import time

# Load the English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Client
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_CHAT_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
)

# List of items to fetch reviews
company_urls = {
    "ASDA": "https://uk.trustpilot.com/review/www.asda.com"
}


# Anonymize the text by replacing entities with placeholders using spacy
#
def anonymize_text(text):
    # Process the text
    doc = nlp(text)
    
    # Anonymized text in parts
    anonymized_parts = []
    last_end = 0 # Track end of the last entity

    for ent in doc.ents:
        anonymized_parts.append(text[last_end:ent.start_char])

        # Use a placeholder based on the type of entity
        if ent.label_ in {"PERSON"}:
            placeholder = "[PERSON]"
        elif ent.label_ in {"ORG"}:
            # Removing the ORG placeholder as it is not required
            #
            # Replace with if required
            # placeholder = "[ORGANIZATION]"
            placeholder = text[ent.start_char:ent.end_char]
        elif ent.label_ in {"GPE", "LOC"}:
            placeholder = "[LOCATION]"
        elif ent.label_ in {"DATE"}:
            placeholder = "[DATE]"
        elif ent.label_ in {"EMAIL"}:
            placeholder = "[EMAIL]"
        else:
            placeholder = "[ANONYMIZED]"  # Generic placeholder for other types

        # Append the placeholder
        anonymized_parts.append(placeholder)

        # Update the last processed character
        last_end = ent.end_char

    # Append the remaining part of the text
    anonymized_parts.append(text[last_end:])

    return ''.join(anonymized_parts)


# Function to generate the sentiment prompt for the user and outcomes from the model
#
def sentiment_promt(user_input):
    messages=[
        {"role": "system", "content": "Classify the sentiment of the following review as positive, negative, or neutral from the user input. The response must be only one of the following: 'positive', 'negative', or 'neutral'. And follow the rules - No additonal information is required. Do not add any empty space before or after the response. Do not add any punctuation marks. Do not add any specical characters. Do not add any extra words. Do not add any extra characters. Do not add any extra spaces. Do not add any extra lines. Do not add any extra paragraphs. Do not add any extra sentences. Do not add any extra symbols. Do not add any extra numbers. Do not add any extra letters. Do not add any extra digits. Do not add any extra alphabets. Do not add any extra digits"},
        {"role": "user", "content": user_input}
    ]
    
    return messages


# Function to generate the topic prompt for the user and outcomes from the model
#
def topic_promt(user_input):
    messages=[
        {"role": "system", "content": "Classify the topic of the user input. Strickly choose on of the topic from the list based on your detailed context matching. The response must be only one of the following: 'in-person shopping', 'delivery experience', 'customer service', 'price', 'product defects and quality'. And follow the rules - No additonal information is required. Do not add any empty space before or after the response. Do not add any punctuation marks. Do not add any specical characters. Do not add any extra words. Do not add any extra characters. Do not add any extra spaces. Do not add any extra lines. Do not add any extra paragraphs. Do not add any extra sentences. Do not add any extra symbols. Do not add any extra numbers. Do not add any extra letters. Do not add any extra digits. Do not add any extra alphabets. Do not add any extra digits"},
        {"role": "user", "content": user_input}
    ]
    
    return messages


# Function to generate the dashboard topic prompt for the user and outcomes from the model
#
def dashboard_topic_promt(user_input):
    messages=[
        {"role": "system", "content": "Classify the topic of the user input. Strickly choose on of the topic from the list based on your detailed context matching. The response must be only one of the following: 'onboarding and setup' ( based on How good is the info to help me get set up? and How easy is it to set up my account?), 'experience' ( based on How easy is it to use? Are the support materials good? Can I get help easily?). And follow the rules - No additonal information is required. Do not add any empty space before or after the response. Do not add any punctuation marks. Do not add any specical characters. Do not add any extra words. Do not add any extra characters. Do not add any extra spaces. Do not add any extra lines. Do not add any extra paragraphs. Do not add any extra sentences. Do not add any extra symbols. Do not add any extra numbers. Do not add any extra letters. Do not add any extra digits. Do not add any extra alphabets. Do not add any extra digits"},
        {"role": "user", "content": user_input}
    ]

    return messages


# Function to execute the LLM model
#
def execute_llm(messages):
    try:
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            messages=messages,
            temperature=0.1,
            max_tokens=60,
            n=1,
        )
        outcome = response.choices[0].message.content.strip().lower()

    except Exception as e:
        print(f"Error: {e}")
        outcome = "unknown" 

    return outcome


# Function to fetch reviews from a given URL
#
def fetch_reviews(base_url):
    """Fetch reviews from a given URL."""
    results = []
    page_number = 1

    while True:
        print(f"page: {page_number}")

        url = f"{base_url}?page={page_number}"
        response = requests.get(url)

        if not response.ok:
            break

        # Only fetch reviews from the first two pages for testing
        # Remove this condition to fetch all reviews
        #
        if page_number == 100:
            break

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        review_script_tag = soup.find('script', {'data-business-unit-json-ld': 'true'})
        if not review_script_tag:
            print("No review data found in the script tag.")
            break

        review_data = json.loads(review_script_tag.string)["@graph"]
        for review in review_data:
            if review["@type"] == "Review":

                url = review.get('@id')

                if not url:
                    generated_uuid = str(uuid.uuid4())
                    review_id = generated_uuid
                else:
                    identifier = url.split('/')[-1]
                    review_id = identifier

                headline = review.get("headline", "No Title")
                review_body = review.get("reviewBody", "No Review Body")
                timestamp = review.get("datePublished", "2000-01-24T14:37:07.000Z")
                rating = review['reviewRating'].get('ratingValue', 'No Rating')
                sentiment = execute_llm(sentiment_promt(f"{headline}. {review_body}"))
                topic = execute_llm(topic_promt(f"{headline}. {review_body}"))
                #dashboard_topic = execute_llm(dashboard_topic_promt(f"{headline}. {review_body}"))
                #"dashboard_topic": dashboard_topic
                results.append({
                    "id": review_id,
                    "title": anonymize_text(headline),
                    "body": anonymize_text(review_body),
                    "timestamp": timestamp,
                    "rating": rating,
                    "page": page_number,
                    "sentiment": sentiment,
                    "topic": topic
                })

        # Uncomment the following line to slow down the requests
        # Continues fetching results empty response
        #
        time.sleep(15)
        page_number += 1

    return results


# Function to load reviews for all company URLs
#
def load_reviews():
    if not os.path.exists('reviews'):
        os.makedirs('reviews')

    for company_name, url in company_urls.items():
        existing_reviews = []

        print(f"Fetching reviews for {company_name}")
        new_reviews = fetch_reviews(url)
        filename = f"reviews/{company_name.replace(' ', '_').lower()}.json"

        existing_reviews = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_reviews = json.load(f)
        
        existing_reviews.extend(new_reviews)

        with open(filename, 'w') as f:
            json.dump(existing_reviews, f, indent=4)
        print(f"Saved reviews for {company_name} to {filename}.")


# Main function
if __name__ == "__main__":
    load_reviews()

