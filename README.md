# Belum di edit pada, bagian ini
def cross_validation(n_splits, model, X, y):
    acc_auc = 0 
    precision = 0
    recall = 0
    f1_score = 0
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_val_fold)
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val_fold)
            # If multi label, set multi class to "ovo", other wise if not multi class use default
            roc_acc = roc_auc_score(y_val_fold, y_pred_proba, multi_class="ovo")
        else:
            print(f"Model {model} tidak mendukung predict_proba, skip ROC AUC.")
            roc_acc = 0 

        print(f"--- Fold {fold+1} ---")
        report_dict = classification_report(y_val_fold, y_pred, output_dict=True, zero_division=0)
        
        acc_auc += roc_acc
        precision += report_dict["macro avg"]["precision"]
        recall += report_dict["macro avg"]["recall"]
        f1_score += report_dict["macro avg"]["f1-score"]
        
        print(f"ROC AUC for fold {fold+1} : {roc_acc:.4f}")

    avg_auc = acc_auc / n_splits
    avg_precision = precision / n_splits
    avg_recall = recall / n_splits
    avg_f1 = f1_score / n_splits

    print("="*30)
    print(f"Average ROC AUC      : {avg_auc:.4f}")
    print(f"Average Precision    : {avg_precision:.4f}")
    print(f"Average Recall       : {avg_recall:.4f}")
    print(f"Average F1 Score     : {avg_f1:.4f}")
    
    return model, avg_auc, avg_precision, avg_recall, avg_f1

# Cross validation RandomForestClassifier model
model = RandomForestClassifier()
rf_model, rf_acc, rf_precision, rf_recall, rf_f1_score = cross_validation(5, model, X_train_scaled, y_train.values)
dan seterusnya
...
# Gabungkan semua hasil dari masing-masing model evaluasi sebelumnya kedalam pandas dataframe
pd.DataFrame({
    "accuracy": [rf_acc, lr_acc, svc_acc, nb_acc, kn_acc, dt_acc], 
    "precision": [rf_precision, lr_precision, svc_precision, nb_precision, kn_precision, dt_precision],
    "recall": [rf_recall, lr_recall, svc_recall, nb_recall, kn_recall, dt_recall],
    "f1_score": [rf_f1_score, lr_f1_score, svc_f1_score, nb_f1_score, kn_f1_score, dt_f1_score]
}, index=["Random Forest", "Logistic Regression", "SVC", "Naive Bayes", "KNeighbors", "Decision Tree"])


report_dict = classification_report(y_test, y_pred_grid, output_dict=True)
grid_acc = report_dict["accuracy"]
grid_precision = report_dict["macro avg"]["precision"]
grid_recall = report_dict["macro avg"]["recall"]
grid_f1_score = report_dict["macro avg"]["f1-score"]


pd.DataFrame({
    "accuracy": [rf_acc, lr_acc, svc_acc, nb_acc, kn_acc, dt_acc, grid_acc], 
    "precision": [rf_precision, lr_precision, svc_precision, nb_precision, kn_precision, dt_precision, grid_precision],
    "recall": [rf_recall, lr_recall, svc_recall, nb_recall, kn_recall, dt_recall, grid_recall],
    "f1_score": [rf_f1_score, lr_f1_score, svc_f1_score, nb_f1_score, kn_f1_score, dt_f1_score, grid_f1_score]
}, index=["Random Forest", "Logistic Regression", "SVC", "Naive Bayes", "KNeighbors", "Decision Tree", "Grid Random Forest"])



# Description
Bank Account Fraud Detection is a machine learning project designed to identify and predict suspicious or fraudulent activity during account openings and banking transactions. In the financial industry, losses due to fraud are significant. This project aims to build a predictive model that can automatically distinguish between legitimate customers/transactions and those with potential for fraud based on historical data patterns, thereby enhancing bank security systems and preventing financial losses.

# Conclusion
- In the dataset used, which contains 1 million rows of data, there is a difference in the amount of data 
between positive and negative classes in the target variable, with the negative class having more 
data than the positive class. This difference has a ratio of 90:10, so 
a slicing technique was applied to the negative class to balance the amount of data. The researcher 
did not use the SMOTE technique because the ratio difference was 
quite significant. The concept of the SMOTE technique involves replicating a specific class, 
which means duplicating the data, and this could lead to 
the model becoming overfitted.
- The feature extraction technique used is SelectKBest with the 5 best features, 
including month, velocity_4w, velocity_24h, housing_status, and 
credit_risk_score. There are outliers in credit_risk_score and housing status, 
so outlier removal is performed on these features.
- Deep learning methods are superior to classical machine learning because 
deep learning methods are generally used for complex tasks, ranging from 
structured data to unstructured data.


For recommendations from researchers, it is suggested to balance the amount of 
uniform data in each target variable class, namely the fraud_bool feature, to 
obtain  optimal  results  and  reduce  the  occurrence  of  model overfitting.  Due to 
the scope and limitations of the research, it is recommended to perform hyperparameter 
tuning again on the deep learning method to achieve more optimal results. Hyperparameter 
tuning can be performed on the number of layers, neurons, and activation function settings
such as ReLu and so on.
