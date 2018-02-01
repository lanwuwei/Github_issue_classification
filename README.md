# Github_issue_classification
1. This is a simple Logistic Regression model for Github issue classification. The feature we used here is bag-of-words.

2. Currently we consider two labels: 'bug' and 'enhancement', we merge subclass into a larger class, for example we treat
'bug-repositiry' label as 'bug' label.

3. dev_merged_annotation.json in ./data folder contains 201 human annotated examples with json format. 

4. The model is trained on 23k labeled examples in train.txt or 230k labeled examples in train_merged.txt (check this link https://drive.google.com/drive/folders/1K-rh9dpaiHJv2GOq73_b_silmw-lrW0S?usp=sharing to download and put it in ./data folder) and evaluated on 201 human annotated labels

5. if you want to run this code, make sure you have sklearn, matplotlib and other libraries installed. You also need to specify the following parameters:

    --tag: bug or enhancement
    
    --use_unlabeled_data: True or False
    
    --use_saved_model: True or False
    
    
    example command to run:
    
    python main.py --tag bug --use_unlabeled_data False --use_saved_model False
