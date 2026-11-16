import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading data...")
    COLS = ['age','workclass','fnlwgt','education','education-num','marital-status',
            'occupation','relationship','race','sex','capital-gain','capital-loss',
            'hours-per-week','native-country','income']
    
    df = pd.read_csv('data/adult.data', names=COLS, skipinitialspace=True)
    df = df.replace('?', np.nan)
    df['income_binary'] = (df['income'] == '>50K').astype(int)

    # Drop redundant or unhelpful features (decided in EDA)
    X = df.drop(columns=['fnlwgt', 'education', 'income', 'income_binary'])
    y = df['income_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline...")
    num_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=42, n_jobs=-1
        ))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Saving model to model.bin...")
    joblib.dump(pipeline, 'model.bin')
    print("Done! Model trained and saved.")

if __name__ == "__main__":
    main()