import json
from dotenv import load_dotenv
from langchain_openai import OpenAI
import pandas as pd
import time
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the language model
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

def analyze_review(review_text, max_retries=3):
    prompt = f"""
    Please analyze the sentiment of this review: '{review_text}' and classify it as Positive, Neutral, or Negative with a rating (0 - 5).
    Also identify if any of these categories [Cleanliness, Staff Behaviour / Hospitality, Room Quality / Comfort, Food & Beverage, Location, Facilities / Amenities, Value for Money, Check-in / Check-out Process, Safety & Security, Noise Levels, Maintenance] are mentioned.
    Only for each mentioned category, classify the sentiment as Positive, Neutral, or Negative.
    Only give the output in the format as JSON with keys 'classify', 'rating', and a key for each identified category with its sentiment.
    If none of the categories are mentioned, only include the 'classify' and 'rating' keys in the output.
    The data types of the values should be string and int respectively.
    Example output: {{
        "classify": "Positive", 
        "rating": 4,
        "Cleanliness": "Positive",
        "Staff Behaviour / Hospitality": "Neutral"
    }} or {{
        "classify": "Neutral",
        "rating": 3
    }}
    """
    
    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        print('#####################################################')
        print('Review: ',review_text)
        print(response)
        
        try:
            response_json = json.loads(response)
            return response_json
        except json.JSONDecodeError:
            print(f"JSON decoding failed on attempt {attempt + 1}. Retrying...")
            time.sleep(1)  # Adding a small delay before retrying
    
    # If all attempts fail, return a default response
    return {"classify": "Neutral", "rating": 0}

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
    
    # Ensure 'Merit' column exists in df_merit
    if 'Merit' not in df_merit.columns:
        df_merit['Merit'] = 0

    # Initialize new columns in the DataFrame
    df['classify'] = ""
    df['rating'] = 0

    # Loop over the rows of the DataFrame
    for index, row in df.iterrows():
        review_text = row['reviews_text']
        analysis_data = analyze_review(review_text)
        
        # Update the DataFrame with the results
        df.at[index, 'classify'] = analysis_data['classify']
        df.at[index, 'rating'] = int(analysis_data['rating'])

        # Update the merit points in df_merit
        try:
            if 'positive' in analysis_data['classify'].lower():
                df_merit.loc[df_merit['Hotel'] == 'Metro_Points_Hotel_Washington_North', 'Merit'] += 1
            elif 'negative' in analysis_data['classify'].lower():
                df_merit.loc[df_merit['Hotel'] == 'Metro_Points_Hotel_Washington_North', 'Merit'] -= 1
        except KeyError:
            print("Hotel not found in the Merit CSV")
        
        # Handle additional categories
        categories = [
            "Cleanliness", "Staff Behaviour / Hospitality", "Room Quality / Comfort", "Food & Beverage",
            "Location", "Facilities / Amenities", "Value for Money", "Check-in / Check-out Process",
            "Safety & Security", "Noise Levels", "Maintenance"
        ]

        for category in categories:
            if category in analysis_data:
                df.at[index, category] = analysis_data[category]

    # Save the updated DataFrames to CSV files
    df.to_csv("output.csv", index=False)
    df_merit.to_csv('Merit.csv', index=False)

if __name__ == "__main__":
    main()
