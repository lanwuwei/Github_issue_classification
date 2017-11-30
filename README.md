# Github_issue_classification
1. This is a simple Logistic Regression model for Github issue classification. The feature we used here is bag-of-words.

2. Currently we consider two labels: 'bug' and 'enhancement', we merge subclass into a larger class, for example we treat
'bug-repositiry' label as 'bug' label.

3. All the txt files in data folder contains sampled 'issueEvent' from Archive Github, the data format is json; dev_merged_annotation.json contains
201 human annotated examples.

4. The model is trained on 23523 labeled examples (may or may not include 54k unlabeled examples) in train.txt and evaluated on 201 human annotated labels

5. if you want to run this code, you need to specify the following four parameters in Github_Issue_Classifier:
    data_folder='data/'
    label_list=['bug','others'] or ['enhancement', 'others']
    saved_model='saved_model.sav'
    use_unlabeled_data=True or False
