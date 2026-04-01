import os
import pandas as pd

"""
    Load specific datasets with his name and filepath.

    Datasets:
    - mMTC: Massive Machine Type Communications
    - URLLC: Ultra-Reliable Low Latency Communications
    - eMBB: Enhanced Mobile Broadband
    - TON_IoT: Real-world IoT dataset (attack detection)
"""

def load_dataset(name, filepath):
    if os.path.exists(filepath):
        # Detect separator
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline()

        sep = ';' if first_line.count(';') > first_line.count(',') else ','
        print(f'{name}: detected separator={repr(sep)}')

        df = pd.read_csv(filepath, sep=sep, low_memory=False)

        if name == 'TON_IoT':
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            df.columns = df.columns.str.strip()

            if 'Label' in df.columns:
                df = df.rename(columns={'Label': 'label_raw'})

            df = df[pd.to_numeric(df['label_raw'], errors='coerce').notna()]
            df['label_raw'] = df['label_raw'].astype(float)

            df['Label'] = df['label_raw'].astype(int).map({
                0: 'Benign',
                1: 'Malicious'
            })
        else:
            df.columns = df.columns.str.strip()

        print(f'Loaded {name}: {df.shape}')
        return df

    else:
        print(f'Skipped {name}: file not found ({filepath})')
        return None

"""
    Load all datasets used in the project.

    Datasets:
    - mMTC: Massive Machine Type Communications
    - URLLC: Ultra-Reliable Low Latency Communications
    - eMBB: Enhanced Mobile Broadband
    - TON_IoT: Real-world IoT dataset (attack detection)
"""

def load_all_datasets():
    datasets = {}

    base_path = os.path.join(os.path.dirname(__file__), "..", "data")

    datasets['mMTC'] = load_dataset(
        'mMTC',
        os.path.join(base_path, "Data5G", "mMTC.csv")
    )

    datasets['URLLC'] = load_dataset(
        'URLLC',
        os.path.join(base_path, "Data5G", "URLLC.csv")
    )

    datasets['eMBB'] = load_dataset(
        'eMBB',
        os.path.join(base_path, "Data5G", "eMBB.csv")
    )

    datasets['TON_IoT'] = load_dataset(
        'TON_IoT',
        os.path.join(base_path, "Data6G", "train_test_network.csv")
    )

    # Remove None datasets
    datasets = {k: v for k, v in datasets.items() if v is not None}

    print(f'\nLoaded {len(datasets)} datasets:', list(datasets.keys()))

    return datasets