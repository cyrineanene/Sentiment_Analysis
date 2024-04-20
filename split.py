import pandas as pd

def balance_dataset(data, n_samples_per_class=20000):
    
    balanced_data = pd.DataFrame()

    for category in data['review/score'].unique():
        subset = data[data['review/score'] == category].sample(n=n_samples_per_class, random_state=42, replace=True)
        balanced_data = pd.concat([balanced_data, subset], axis=0)

    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_data


df = pd.read_csv('./BR3.csv')  
balanced_df = balance_dataset(df)
balanced_df.to_csv('balanced_dataset2.csv', index=False) 
