import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
from openai import OpenAI
import os
import json

with open('openai_key.json', 'r') as file:
    data = json.load(file)
api_key = data['API_KEY']
#print(api_key)
client = OpenAI(api_key=api_key)

def standardize_data_keys(data):
    """
    Standardize the keys of dictionaries in the list `data` to match the expected
    keys by the DataTable, handling different naming conventions and case sensitivity.
    """
    standardized_data = []
    key_mapping = {
        'name': ['Name', 'location', 'Location', 'place', 'Place', 'locations', 'Locations', 'places', 'Places'],
        'x': ['X Coordinate', 'x coordinate', 'X', 'x'],
        'y': ['Y Coordinate', 'y coordinate', 'Y', 'y'],
        'population': ['Population', 'population'],
        'infection_source': ['Infection Source', 'infection source', 'InfectionSource'],
        'vaccination_rate': ['Vaccination Rate', 'vaccination rate', 'VaccinationRate'],
        'mobility': ['Mobility', 'mobility']
    }
    
    inverted_mapping = {v.lower(): k for k, vals in key_mapping.items() for v in vals}
    
    for row in data:
        standardized_row = {}
        for key, value in row.items():
            standardized_key = inverted_mapping.get(key.lower(), key)  
            standardized_row[standardized_key] = value
        standardized_data.append(standardized_row)
    
    return standardized_data

def add_detailed_descriptions(df):
    previous_row = None
    previous_node = None
    
    def get_detailed_description(current_row, previous_row):
        if previous_row is None or current_row['Node'] != previous_row['Node']:
            return "Initial data point for node."
        
        desc = []
        
        if current_row['Infected'] > previous_row['Infected']:
            if current_row['Infected'] > previous_row['Recovered']:
                desc.append("Infection rising sharply.")
            else:
                desc.append("Infection increasing but recovery is higher.")
        elif current_row['Infected'] < previous_row['Infected']:
            desc.append("Infection declining.")
        
        if current_row['Recovered'] > previous_row['Recovered']:
            desc.append("Recovery numbers improving.")
        
        if current_row['Susceptible'] < previous_row['Susceptible']:
            desc.append("Susceptible population decreasing.")
        
        if not desc: 
            return "Stable condition with no significant changes from previous timestep."
        
        return ' '.join(desc)
    
    descriptions = []
    for index, row in df.iterrows():
        description = get_detailed_description(row, previous_row if previous_node == row['Node'] else None)
        descriptions.append(description)
        previous_row = row
        previous_node = row['Node']
    
    df['Description'] = descriptions
    return df

def advanced_filter_data_based_on_query(query):
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    df_descriptions = pd.read_excel('output_with_descriptions.xlsx')
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stop_words]

    filter_mask = pd.Series([False] * len(df_descriptions))

    node_regex = '|'.join(map(re.escape, filtered_words))
    filter_mask |= df_descriptions['Node'].str.contains(node_regex, case=False, regex=True)

    conditions = ['rising', 'declining', 'improving', 'decreasing', 'stable']
    matched_conditions = [word for word in filtered_words if word in conditions]
    for condition in matched_conditions:
        if condition in ['rising', 'increasing']:
            filter_mask |= df_descriptions['Description'].str.contains('rising|increasing', case=False)
        elif condition in ['declining', 'decreasing']:
            filter_mask |= df_descriptions['Description'].str.contains('declining|decreasing', case=False)
        elif condition == 'improving':
            filter_mask |= df_descriptions['Description'].str.contains('improving', case=False)
        elif condition == 'stable':
            filter_mask |= df_descriptions['Description'].str.contains('stable', case=False)

    filtered_df = df_descriptions[filter_mask]
    if filtered_df.empty:
        return "No data matching your query was found."
    return filtered_df

def get_middle_description(s):
    n = len(s)
    middle_index = n // 2
    if n % 2 == 0:
        middle_index -= 1
    sorted_s = s.sort_values() 
    return sorted_s.iloc[middle_index]

def summarize_data_by_time_and_node(data_df, interval=25):
    data_df['Time Interval'] = (data_df['Time'] // interval) * interval
    
    summary_df = data_df.groupby(['Time Interval', 'Node']).agg({
        'Susceptible': 'mean',
        'Infected': 'mean',
        'Recovered': 'mean',
        'Description': get_middle_description  
    }).reset_index()

    summary_texts = []
    for _, row in summary_df.iterrows():
        summary = f"At time {row['Time Interval']} to {row['Time Interval'] + interval}, node {row['Node']}: {int(row['Susceptible'])} susceptible, {int(row['Infected'])} infected, and {int(row['Recovered'])} recovered. Description: {row['Description']}"
        summary_texts.append(summary)
    return " ".join(summary_texts)


def convert_data_to_text(data_df):
    if isinstance(data_df, str):  
        return data_df
    
    summary_texts = []
    for _, row in data_df.iterrows():
        summary = f"At time {row['Time']}, in node {row['Node']}, there were {row['Susceptible']} susceptible, {row['Infected']} infected, and {row['Recovered']} recovered. Description: {row['Description']}"
        summary_texts.append(summary)
    return " ".join(summary_texts)

def ask_openai(query, previous_messages):
    filtered_df = advanced_filter_data_based_on_query(query)
    if isinstance(filtered_df, str) and "No data matching" in filtered_df:
        messages = previous_messages + [{"role": "user", "content": query}]
    else:
        summarized_df = summarize_data_by_time_and_node(filtered_df)
        text_for_ai = convert_data_to_text(summarized_df)
        print("Text for AI:", text_for_ai)
        
        messages = previous_messages + [
            {"role": "system", "content": "You are an AI that assists with understanding simulation data. Answer questions based on the data. Ensure that your answer is detailed with numbers and timing based on the data. Only pick clear and important patterns. Do not provide vague or irrelevant information."},
            {"role": "user", "content": text_for_ai},
            {"role": "user", "content": query}
        ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    
    if response.choices:
        ai_response = response.choices[0].message.content.strip()
    else:
        ai_response = "No response generated."

    previous_messages.append({"role": "user", "content": query})
    previous_messages.append({"role": "assistant", "content": ai_response})

    return ai_response, previous_messages