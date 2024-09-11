import pandas as pd
from tqdm import tqdm

train_df = pd.read_csv("train.csv", index_col=False)
state_df = pd.read_csv("state_labels.csv", index_col=False)
color_df = pd.read_csv("color_labels.csv", index_col=False)
breed_df = pd.read_csv("breed_labels.csv", index_col=False)

# Task: Predict AdoptionSpeed in [0, 1, 2, 3, 4]

from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

description_list = []
for index, row in tqdm(train_df.iterrows(), total = len(train_df)):
    # Type
    train_df['Type'] = train_df['Type'].astype(str)
    train_df.loc[index, 'Type'] = "Dog" if row['Type'] == '1' else "Cat"

    # Breed1
    train_df['Breed1'] = train_df['Breed1'].astype(str)
    train_df.loc[index, 'Breed1'] = breed_df[breed_df['BreedID']==row['Breed1']]['BreedName'].values[0] if row['Breed1'] != 0 else ''

    # Breed2
    train_df['Breed2'] = train_df['Breed2'].astype(str)
    train_df.loc[index, 'Breed2'] = breed_df[breed_df['BreedID']==row['Breed2']]['BreedName'].values[0] if row['Breed2'] != 0 else ''

    # Gender
    train_df['Gender'] = train_df['Gender'].astype(str)
    train_df.loc[index, 'Gender'] = {1 : 'Male', 2 : 'Female', 3 : 'Mixed'}[row['Gender']]

    # Color1
    train_df['Color1'] = train_df['Color1'].astype(str)
    train_df.loc[index, 'Color1'] = color_df[color_df['ColorID']==row['Color1']]['ColorName'].values[0] if row['Color1'] != 0 else ''

    # Color2
    train_df['Color2'] = train_df['Color2'].astype(str)
    train_df.loc[index, 'Color2'] = color_df[color_df['ColorID']==row['Color2']]['ColorName'].values[0] if row['Color2'] != 0 else ''

    # Color3
    train_df['Color3'] = train_df['Color3'].astype(str)
    train_df.loc[index, 'Color3'] = color_df[color_df['ColorID']==row['Color3']]['ColorName'].values[0] if row['Color3'] != 0 else ''

    # MaturitySize
    train_df['MaturitySize'] = train_df['MaturitySize'].astype(str)
    train_df.loc[index, 'MaturitySize'] = {1 : 'Small', 2 : 'Medium', 3 : 'Large', 4 : 'Extra Large', 0 : 'Not Specified'}[row['MaturitySize']]

    # FurLength
    train_df['FurLength'] = train_df['FurLength'].astype(str)
    train_df.loc[index, 'FurLength'] = {1 : 'Short', 2 : 'Medium', 3 : 'Long', 0 : 'Not Specified'}[row['FurLength']]

    # Vaccinated
    train_df['Vaccinated'] = train_df['Vaccinated'].astype(str)
    train_df.loc[index, 'Vaccinated'] = {1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}[row['Vaccinated']]

    # Dewormed
    train_df['Dewormed'] = train_df['Dewormed'].astype(str)
    train_df.loc[index, 'Dewormed'] = {1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}[row['Dewormed']]

    # Sterilized
    train_df['Sterilized'] = train_df['Sterilized'].astype(str)
    train_df.loc[index, 'Sterilized'] = {1 : 'Yes', 2 : 'No', 3 : 'Not Sure'}[row['Sterilized']]

    # Health
    train_df['Health'] = train_df['Health'].astype(str)
    train_df.loc[index, 'Health'] = {1 : 'Healthy', 2 : 'Minor Injury', 3 : 'Serious Injury', 0 : 'Not Specified'}[row['Health']]

    # State
    train_df['State'] = train_df['State'].astype(str)
    train_df.loc[index, 'State'] = state_df[state_df['StateID']==row['State']]['StateName'].values[0] if row['State'] != 0 else ''

    # Description
    if pd.notna(row['Description']) and 0<len(row['Description']) and len(row['Description']) < 5000:
        description_list.append(row['Description'])
    else:
        train_df.loc[index, 'Description'] = 'None'


trans_description_list = translator.translate_batch(description_list)
trans_dict = dict(zip(description_list, trans_description_list))

for index, row in tqdm(train_df.iterrows(), total = len(train_df)):
    if row['Description'] in trans_dict:
        train_df.loc[index, 'Description'] = trans_dict[row['Description']]

train_df.to_csv('processed_train.csv', index = False)