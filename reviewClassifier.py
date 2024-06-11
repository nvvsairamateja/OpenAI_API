import json
from dotenv import load_dotenv
from langchain_openai import OpenAI
import pandas as pd
import time
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(openai_api_key = OPENAI_API_KEY )


def analyze_review(review_text, max_retries=3):
    prompt = f"""
    Please analyze the sentiment of this review: '{review_text}' and classify it as Positive, Neutral, or Negative with a confidence score (0 - 1) and rating (0 - 5).
    Only give the output in the format as JSON with keys 'classify', 'confidence_score', and 'rating'.
    The data types of the values should be string, float, and int respectively.
    """
    prompt = prompt + 'Example output: {"classify": "Positive", "confidence_score": 0.85, "rating": 4}'
    
    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        print('#####################################################')
        print(response)
        
        try:
            response_json = json.loads(response)
            return response_json
        except json.JSONDecodeError:
            print(f"JSON decoding failed on attempt {attempt + 1}. Retrying...")
            time.sleep(1)  # Adding a small delay before retrying
    
    # If all attempts fail, return a default response
    return {"classify": "Neutral", "confidence_score": 0.0, "rating": 0}

def main():
    try:
        # Load the DataFrame
        df = pd.read_csv("Metro_Points_Hotel_Washington_North.csv").sample(20)
        df_merit = pd.read_csv('Merit.csv')
    except FileNotFoundError:
        print("The CSV file was not found.")
        return
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error parsing the CSV file.")
        return
    
    # Initialize new columns in the DataFrame
    df['classify'] = ""
    df['confidence_score'] = 0.0
    df['rating'] = 0

    for index, row in df.iterrows():
        review_text = row['reviews_text']
        analysis_data = analyze_review(review_text)
        # Update the DataFrame with the results
        df.at[index, 'classify'] = analysis_data['classify']
        df.at[index, 'confidence_score'] = analysis_data['confidence_score']
        df.at[index, 'rating'] = int(analysis_data['rating'])

        try:
            if 'positive' in analysis_data['classify'].lower():
                df_merit.loc[df_merit['Hotel'] == 'Metro_Points_Hotel_Washington_North', 'Merit'] += 1
            elif 'negative' in analysis_data['classify'].lower():
                df_merit.loc[df_merit['Hotel'] == 'Metro_Points_Hotel_Washington_North', 'Merit'] -= 1
        except KeyError:
            print("Hotel not found in the Merit CSV")

    df.to_csv("output.csv", index=False)
    df_merit.to_csv('Merit.csv', index=False)

if __name__ == "__main__":
    main()