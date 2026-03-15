import pandas as pd
import requests
import os
from sklearn.preprocessing import LabelEncoder

def download_german_credit(save_dir='data'):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    filename = os.path.join(save_dir, 'german.data')
    os.makedirs(save_dir, exist_ok=True)

    # Download if not exists
    if not os.path.exists(filename):
        print("Downloading German Credit dataset...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    else:
        print("File already exists, loading locally.")

    # Column names according to UCI documentation
    columns = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_status', 'employment', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
        'housing', 'num_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'class'
    ]

    df = pd.read_csv(filename, sep=' ', header=None, names=columns)

    # Target: class 1=good, 2=bad -> convert to 1/0 (1=good, 0=bad)
    df['class'] = df['class'].replace({1: 1, 2: 0})

    # Create sensitive attribute: gender from personal_status_sex
    # A91: male (divorced/separated), A92: female (divorced/separated/married),
    # A93: male (single), A94: male (married/widowed), A95: female (single)
    df['gender'] = df['personal_status_sex'].apply(
        lambda x: 1 if x in ['A91', 'A93', 'A94'] else 0   # 1 = male, 0 = female
    )

    # Drop original personal_status_sex
    df = df.drop(columns=['personal_status_sex'])

    # One-hot encode categorical variables (simplified: convert to codes for simplicity)
    # For a full proper encoding, we would use pd.get_dummies, but here we keep it simple
    # We'll convert categorical columns to category codes
    categorical_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status',
                        'employment', 'other_debtors', 'property', 'other_installment_plans',
                        'housing', 'job', 'telephone', 'foreign_worker']

    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes

    # Save processed data
    output_file = os.path.join(save_dir, 'german_credit_processed.csv')
    df.to_csv(output_file, index=False)
    print(f"Processed German Credit saved to {output_file}")

    return df

if __name__ == '__main__':
    download_german_credit()