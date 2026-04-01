import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

log1p_transformer = FunctionTransformer(np.log1p, validate=False)

# Columns that are never features — identifiers, leakage, raw labels
DROP_COLS = [
    'Label', 'label_raw', 'predicted', 'UniqueID', 'X',
    'anomaly_type', 'type',
    'SrcAddr', 'DstAddr', 'Sport', 'Dport',
    'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'source_ip', 'destination_ip', 'timestamp'
]

def make_xy(df, label_col='Label'):
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop, errors='ignore')
    y = df[label_col]
    return X, y

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p',   log1p_transformer),
        ('scaler',  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    transformers = []
    if num_cols: transformers.append(('num', num_pipe, num_cols))
    if cat_cols: transformers.append(('cat', cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder='drop')