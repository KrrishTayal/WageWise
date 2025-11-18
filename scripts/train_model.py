import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json


def preprocess_data(df):
    top_skills = [
        'Python', 'SQL', 'Machine Learning', 'AWS', 'JavaScript', 
        'Financial Modeling', 'Project Management', 'Leadership'
    ]
    for skill in top_skills:
        df[f'skill_{skill.lower().replace(" ", "_")}'] = df['skills'].apply(lambda x: 1 if skill in x else 0)
    return df.drop('skills', axis=1)


df = preprocess_data(pd.read_csv('data/salary_data.csv'))

X = df.drop('salary', axis=1)
y = df['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'experience']
# MODIFIED: Added 'country' to categorical features
categorical_features = ['education', 'job_title', 'location', 'country', 'company_size']
skill_features = [col for col in df.columns if col.startswith('skill_')]

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
    ('skills', 'passthrough', skill_features)
])

models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
}

best_model, best_score, results = None, -np.inf, {}

for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'R2': r2}
    print(f"{name}: MAE = {mae:.2f}, R2 = {r2:.2f}")
    if r2 > best_score:
        best_score, best_model = r2, pipeline

column_order = X_train.columns.tolist()
with open('models/column_order.json', 'w') as f:
    json.dump(column_order, f)

joblib.dump(best_model, 'models/salary_predictor.pkl')
best_model.fit(X_train, y_train)

if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance[importance['feature'].str.startswith('skill_')])