import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Create TD / TF-IDF Matricies
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression # Preform Logistic Regression
from sklearn.metrics import confusion_matrix # Make the Confustion Matrix
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score # for AUC, fpr, tpr, threshold and accuracy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from nltk.corpus import stopwords # for the Stopwords list
from nltk.stem import PorterStemmer # for the porter Stemmer
import nltk # for other nltk functions
import re # for regular expression functions

from tqdm import tqdm  # for progress bar
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('punkt')
nltk.download('stopwords')

!pip install optuna
import optuna

# Read in data
job_postings = pd.read_csv("/kaggle/input/linkedin-job-postings/postings.csv")

# Subsetting For Job Description and Experience Level
want =  ["description","formatted_experience_level"]
df = job_postings[want]
# Dropping Missing Values
df = df.dropna(subset=['formatted_experience_level']).reset_index(drop=True)


df['description'] = df['description'].astype(str)
df["formatted_experience_level"] = np.where(df["formatted_experience_level"] == 'Entry level',1,0)
### Feature Engineering
def Features(df, column_name):
    feature_columns = ['word_cnt', 'sent_cnt', 'vocab_cnt', 'Avg_sent_word_cnt', 'lexical_richness','Readability_index']
    feature_data = []

    for index, row in df.iterrows():
        text = row[column_name]

        # Simple features (Word Count, Sentance Count, Vocabulary Count, Lexical Diversity)
        tokens = nltk.word_tokenize(text)
        char_cnt = len(tokens)

        words = [w for w in tokens if w.isalpha()]
        word_cnt = len(words)

        avg_word_length = char_cnt/word_cnt

        sents = nltk.sent_tokenize(text)
        sent_cnt = len(sents)

        avg_sent_length = word_cnt / sent_cnt if sent_cnt > 0 else 0
        avg_sent_length = round(avg_sent_length,2)

        vocab = set(words)
        vocab_cnt = len(vocab)

        lex_richness = round(vocab_cnt / word_cnt, 4)

        ARI = 4.71*avg_word_length + .5*avg_sent_length - 21.43

        # Append the column data
        feature_data.append([word_cnt, sent_cnt, vocab_cnt, avg_sent_length, lex_richness ,ARI]) # dropped avg_sent_length

    feature_df = pd.DataFrame(feature_data, columns=feature_columns)

    # Combine the original DataFrame with the new DataFrame containing features
    result_df = pd.concat([df, feature_df], axis=1)

    return result_df

### PreProcessing for Feature Engineered Data
FE_df = Features(df,'description')
FE_df['Cust_Service'] = FE_df['description'].apply(lambda x: 1 if 'customer service' in x.lower() else 0)
FE_df['diploma_ged'] = FE_df['description'].apply(lambda x: 1 if 'diploma ged' in x.lower() else 0)
FE_df['per_hour'] = FE_df['description'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
FE_df['diploma_equiv'] = FE_df['description'].apply(lambda x: 1 if 'diploma equivalent' in x.lower() else 0)
FE_df['project_management'] = FE_df['description'].apply(lambda x: 1 if 'project management' in x.lower() else 0)
FE_df['cross_functional'] = FE_df['description'].apply(lambda x: 1 if 'cross functional' in x.lower() else 0)
FE_df['minimum_years'] = FE_df['description'].apply(lambda x: 1 if 'minimum years' in x.lower() else 0)
FE_df['experience_working'] = FE_df['description'].apply(lambda x: 1 if 'experience working' in x.lower() else 0)
FE_df['management'] = FE_df['description'].apply(lambda x: 1 if 'management ' in x.lower() else 0)
FE_df['track_record'] = FE_df['description'].apply(lambda x: 1 if 'track_record ' in x.lower() else 0)
x_fe = FE_df.drop(['description', 'formatted_experience_level'], axis=1)
y_fe = FE_df['formatted_experience_level']

X_train2, X_test2, y_train2, y_test2 = train_test_split(x_fe, y_fe, test_size = 0.2, random_state = 1)
X_train2.head()

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train2, y_train2)

scaler = MinMaxScaler()
X_train_resampled_normalized = scaler.fit_transform(X_train_resampled)
X_test2_normalized = scaler.transform(X_test2)

X_train_resampled_normalized = pd.DataFrame(X_train_resampled_normalized, columns=X_train2.columns)
X_test2_normalized = pd.DataFrame(X_test2_normalized, columns=X_test2.columns)

### Logistic Regression
def objective(trial):

    C_value = trial.suggest_float('C', 1e-4, 1e3, log=True)  # Log-uniform distribution for C

    logistic_fe = LogisticRegression(C=C_value, penalty='l1', solver='liblinear', max_iter=1000)
    logistic_fe_cv_scores = cross_val_score(logistic_fe, X_train_resampled_normalized, y_train_resampled, cv=10, scoring='accuracy')

    return logistic_fe_cv_scores.mean()

start_time = time.time() # record time
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
end_time = time.time() # record time
#Computing Training Duration
logistic_fe_tt = end_time - start_time

print(f"Train Time: {logistic_fe_tt} Seconds")

best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

mean_acc_lfe = study.best_value
print(f"Best 10-Fold CV Accuracy: {mean_acc_lfe:.2%}")

logistic_fe = LogisticRegression(C=best_params['C'], penalty='l1', solver='liblinear', max_iter=1000)
logistic_fe.fit(X_train_resampled_normalized, y_train_resampled)

# Predict on train and test datasets
y_test_pred_lfe = logistic_fe.predict(X_test2_normalized)

#Test Accuracy
test_acc_lfe = accuracy_score(y_test2, y_test_pred_lfe)
print(f"Test Accuracy: {test_acc_lfe:.2%}")

print("Coefficient Weights on Test Data:")
coef_weights_test = logistic_fe.coef_
for feature, coef in zip(X_train_resampled_normalized.columns, coef_weights_test.flatten()):
    print(f"{feature}: {coef:.4f}")

# Compute test probabilities, false positive rate, true positive rate, and auc
y_test_prob_lfe = logistic_fe.predict_proba(X_test2_normalized)[:, 1]  # Probability of class 1 (positive)
fpr_lfe, tpr_lfe, threshold_lfe = roc_curve(y_test2,y_test_prob_lfe)
auc_score_lfe = roc_auc_score(y_test2, y_test_prob_lfe)
precision_lfe, recall_lfe, f1_lfe, _ = precision_recall_fscore_support(y_test2, y_test_pred_lfe)
print(f"AUC Score: {auc_score_lfe:.2%}")
print(f"Precision: {precision_lfe[1]:.2%}")
print(f"Recall: {recall_lfe[1]:.2%}")
print(f"F1-Score: {f1_lfe[1]:.2%}")

# results
# AUC Score: 68.66%
# Precision: 52.04%
# Recall: 66.57%
# F1-Score: 58.41%


### Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB

start_time = time.time()
# Fitting the Naive Bayes model
bayes_fe = GaussianNB()
# Preforming 10-Fold CV
bayes_fe_10fold = cross_val_score(bayes_fe,X_train_resampled_normalized, y_train_resampled, cv=10)
end_time = time.time()
# Computing Train Time
bayes_fe_tt = end_time - start_time
print(f"Training Duration: {bayes_fe_tt} Seconds")

mean_acc_bfe = bayes_fe_10fold.mean()
print(f"10-Fold Cross-Validation Accuracy: {mean_acc_bfe}")


bayes_fe.fit(X_train_resampled_normalized, y_train_resampled)
# Predict on train and test datasets
y_test_pred_bfe = bayes_fe.predict(X_test2_normalized)

#Test Accuracy
test_acc_bfe = accuracy_score(y_test2, y_test_pred_bfe)
print(f"Test Accuracy: {test_acc_bfe:.2%}")

# Compute test probabilities, false positive rate, true positive rate, and auc
y_test_prob_bfe = bayes_fe.predict_proba(X_test2_normalized)[:, 1]  # Probability of class 1 (positive)
fpr_bfe, tpr_bfe, threshold_bfe = roc_curve(y_test2,y_test_prob_bfe)
auc_score_bfe = roc_auc_score(y_test2, y_test_prob_bfe)
precision_bfe, recall_bfe, f1_bfe, _ = precision_recall_fscore_support(y_test2, y_test_pred_bfe)
print(f"AUC Score: {auc_score_bfe:.2%}")
print(f"Precision: {precision_bfe[1]:.2%}")
print(f"Recall: {recall_bfe[1]:.2%}")
print(f"F1-Score: {f1_bfe[1]:.2%}")

#AUC Score: 66.13%
#Precision: 66.37%
#Recall: 6.93%
#F1-Score: 12.55%

### KNN
k_values = list(range(1, 20))
cv_scores_fe = []

start_time_knn = time.time()

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_resampled_normalized, y_train_resampled, cv=10)
    cv_scores_fe.append(scores.mean())

end_time_knn = time.time()
knn_tt_fe1 = end_time_knn - start_time_knn

errors_fe = [(1-i) for i in cv_scores_fe]

figsize=(12, 6)
plt.plot(k_values, errors_fe, marker='o')
plt.title('KNN Error vs. Number of Neighbors (k)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('10 Fold CV Error')
plt.show()

# Find the optimal k with the highest accuracy
optimal_k_fe = k_values[errors_fe.index(min(errors_fe))]
print(f"Optimal K: {optimal_k_fe}")

knn_fe = KNeighborsClassifier(n_neighbors=optimal_k_fe)
start_time_knn = time.time()
knn_10fold_fe = cross_val_score(knn_fe, X_train_resampled_normalized, y_train_resampled, cv=10)
knn_fe.fit(X_train_resampled_normalized, y_train_resampled)
end_time_knn = time.time()

# Computing Training Duration
knn_tt_fe = (end_time_knn - start_time_knn) + knn_tt_fe1
print(f"Training Duration: {knn_tt_fe} Seconds")

mean_acc_knnfe = knn_10fold_fe.mean()
print(f"10-Fold Cross-Validation Accuracy: {mean_acc_knnfe:.2%}")

#Training Duration: 487.02689814567566 Seconds
#10-Fold Cross-Validation Accuracy: 81.09%

# Predict on train and test datasets
y_test_pred_knn_fe = knn_fe.predict(X_test2_normalized)

# Test Accuracy
test_acc_knn_fe = accuracy_score(y_test2, y_test_pred_knn_fe)
print(f"Test Accuracy: {test_acc_knn_fe:.2%}")

# Compute test probabilities, false positive rate, true positive rate, and auc
y_test_prob_knn_fe = knn_fe.predict_proba(X_test2_normalized)[:, 1]  # Probability of class 1 (positive)
fpr_knn_fe, tpr_knn_fe, threshold_knn_fe = roc_curve(y_test2, y_test_prob_knn_fe)
auc_score_knn_fe = roc_auc_score(y_test2, y_test_prob_knn_fe)
precision_knn_fe, recall_knn_fe, f1_knn_fe, _ = precision_recall_fscore_support(y_test2, y_test_pred_knn_fe)
print(f"AUC Score: {auc_score_knn_fe:.2%}")
print(f"Precision: {precision_knn_fe[1]:.2%}")
print(f"Recall: {recall_knn_fe[1]:.2%}")
print(f"F1-Score: {f1_knn_fe[1]:.2%}")
#Test Accuracy: 71.20%
#AUC Score: 69.82%
#Precision: 63.62%
#Recall: 63.26%
#F1-Score: 63.44%


### Random Forest
def objective(trial):
    param_dist = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_features': trial.suggest_categorical('max_features', ['sqrt',None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2,20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,20)
    }

    rf_fe = RandomForestClassifier(criterion='entropy', random_state=42, **param_dist)
    cv_scores = cross_val_score(rf_fe, X_train_resampled_normalized, y_train_resampled, cv=10)

    return cv_scores.mean()

start_time = time.time() # record time
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective, n_trials=5)
end_time = time.time() # record time

#Computing Training Duration
rf_fe_tt = end_time - start_time
print(f"Training Duration: {rf_fe_tt} Seconds")
print("Best Hyperparameters:", study_rf.best_trial.params)
mean_acc_rffe = study_rf.best_value
print(f"10-Fold Cross-Validation Accuracy: {mean_acc_rffe}")
#Training Duration: 5765.755069971085 Seconds
#Best Hyperparameters: {'n_estimators': 326, 'max_features': None, 'min_samples_split': 14, 'min_samples_leaf': 7}
#10-Fold Cross-Validation Accuracy: 0.7859696821921884

# Best model from hyperparameter tuning
rf_fe = RandomForestClassifier(criterion='entropy', random_state=42, **study_rf.best_trial.params)
rf_fe.fit(X_train_resampled_normalized, y_train_resampled)

# Predict on train and test datasets
y_test_pred_rffe = rf_fe.predict(X_test2_normalized)

#Test Accuracy
test_acc_rffe = accuracy_score(y_test2, y_test_pred_rffe)
print(f"Test Accuracy: {test_acc_rffe:.2%}")
#Test Accuracy: 73.19%

# Compute test probabilities, false positive rate, true positive rate, and auc
y_test_prob_rffe = rf_fe.predict_proba(X_test2_normalized)[:, 1]  # Probability of class 1 (positive)
fpr_rffe, tpr_rffe, threshold_rffe = roc_curve(y_test2, y_test_prob_rffe)
auc_score_rffe = roc_auc_score(y_test2, y_test_prob_rffe)
precision_rffe, recall_rffe, f1_rffe, _ = precision_recall_fscore_support(y_test2, y_test_pred_rffe)
print(f"AUC Score: {auc_score_rffe:.2%}")
print(f"Precision: {precision_rffe[1]:.2%}")
print(f"Recall: {recall_rffe[1]:.2%}")
print(f"F1-Score: {f1_rffe[1]:.2%}")

# AUC Score: 79.75%
# Precision: 66.31%
# Recall: 65.28%
# F1-Score: 65.79%
