from __future__ import division
import json
import gzip
import argparse
import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle as pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score

import matplotlib
import matplotlib.pyplot as plt
font = {'size'   : 10,'weight' : 'medium'}
matplotlib.rc('font', **font)
plt.gcf().subplots_adjust(bottom=0.15)
plt.rcParams.update({'axes.labelsize ': 'small'})

class Github_Issue_Classifier():
	def __init__(self, data_folder, label_list, saved_model=None, use_unlabeled_data=False):
		self.data_folder=data_folder
		self.label_list=label_list
		self.saved_model=saved_model
		self.use_unlabeled_data=use_unlabeled_data

	def Feature_Extraction(self, data):
		vectorizer = TfidfVectorizer(stop_words='english',min_df=10,ngram_range=(1, 2))#, min_df=5, ngram_range=(1, 3), strip_accents='ascii')
		#vectorizer = TfidfVectorizer(min_df=5, stop_words='english',ngram_range=(1, 2)) #token_pattern=r'\b\w+\b',
		#vectorizer = HashingVectorizer(stop_words='english', ngram_range=(1, 2))
		X = vectorizer.fit_transform(data)
		features = vectorizer.get_feature_names()
		self.feature_names = features
		counts = X.toarray()
		a=counts[0]
		print('Number of features: '+ str(len(a)))
		return counts

	def Data_Preprocess_gz_file(self):
		list_of_files = []
		for file in os.listdir(self.data_folder):
			if file.endswith(".gz"):
				file_path = (os.path.join(self.data_folder, file))
				list_of_files.append(file_path)
		''''''
		data=[]
		label=[]
		for file in list_of_files:
			with gzip.open(file, 'rb') as json_file:
				for line in json_file:
					tmp = json.loads(line)
					if tmp['type'] == 'IssuesEvent':
						try:
							if tmp['payload']['issue']['labels']:
								label_flag=True
								for element in tmp['payload']['issue']['labels']:
									label_name=element['name'].lower()
									if 'bug' in label_name and 'debug' not in label_name:
										label_name = 'bug'
									elif 'feature' in label_name or 'enhancement' in label_name:
										label_name = 'enhancement'
									if label_name in self.label_list:
										label.append(label_name)
										content = ''
										if tmp['payload']['issue']['title']:
											content = tmp['payload']['issue']['title'].encode('utf-8') + '\t'
										if tmp['payload']['issue']['body']:
											content += tmp['payload']['issue']['body'].encode('utf-8')
										data.append(content)
										label_flag=False
										break # single label currently
									else:
										continue
								if label_flag:
									label.append('others')
									content=''
									if tmp['payload']['issue']['title']:
										content=tmp['payload']['issue']['title'].encode('utf-8') + '\t'
									if tmp['payload']['issue']['body']:
										content+=tmp['payload']['issue']['body'].encode('utf-8')
									data.append(content)
						except:
							print file
							print line
							sys.exit()

		pickle.dump(data, open("data2.p", "wb"))
		pickle.dump(label, open("label2.p", "wb"))
		num_bug=0
		num_enhancement=0
		num_others=0
		for item in label:
			if item=='bug':
				num_bug+=1
			elif item=='enhancement':
				num_enhancement+=1
			else:
				num_others+=1
		print(len(label))
		print(num_bug)
		print(num_enhancement)
		print(num_others)
		''''''
		self.data=pickle.load(open("data2.p", "rb"))
		self.label=pickle.load(open("label2.p","rb"))

		data=self.Feature_Extraction(self.data)
		label=self.label

		# split train, dev and test
		cutoff=len(data) / 2
		train_set=(data[:cutoff],label[:cutoff])
		cutoff_2 = cutoff + cutoff/2
		dev_set = (data[cutoff:cutoff_2],label[cutoff:cutoff_2])
		test_set=(data[cutoff_2:],label[cutoff_2:])
		print(len(train_set[0]))
		print(len(dev_set[0]))
		print(len(test_set[0]))
		print('data split finished...')
		self.test_split=cutoff_2
		return (train_set, dev_set, test_set)

	def Data_Preprocess_txt_file(self):
		list_of_files = []
		for file in os.listdir(self.data_folder):
			if file.endswith(".txt"):
				file_path = (os.path.join(self.data_folder, file))
				list_of_files.append(file_path)
		for file in os.listdir(self.data_folder):
			if file.endswith(".json"):
				annotation_filepath = (os.path.join(self.data_folder, file))
		print('We use evaluation data in this file: '+annotation_filepath)
		gold_data=[]
		gold_label=[]
		gold_id=[]
		automatic_label=[]
		with open(annotation_filepath, 'r') as json_file:
			for line in json_file:
				tmp = json.loads(line)
				'''
				if tmp['payload']['issue']['labels']:
					label_flag = True
					for element in tmp['payload']['issue']['labels']:
						label_name = element['name'].lower()
						if 'bug' in label_name and 'debug' not in label_name:
							label_name = 'bug'
						if 'feature' in label_name or 'enhancement' in label_name:
							label_name = 'enhancement'
						if label_name in 'enhancement' or label_name in 'bug':  # self.label_list:
							automatic_label.append(label_name)
							label_flag = False
							break  # single label currently
						else:
							continue
					if label_flag:
						automatic_label.append('others')
				'''
				if True:
				#if not tmp['payload']['issue']['labels']:
					gold_id.append(tmp['id'])
					content = ''
					if tmp['payload']['issue']['title']:
						content = tmp['payload']['issue']['title'].encode('utf-8') + '\t'
					if tmp['payload']['issue']['body']:
						content += tmp['payload']['issue']['body'].encode('utf-8')
					gold_data.append(content)
					#gold_label_list=tmp['label_annotated'].strip().split(",")
					if self.label_list[0] in tmp['label_annotated']:# or 'bug' in tmp['label_annotated']:
						#gold_label.append('enhancement')
						gold_label.append(True)
					else:
						gold_label.append(False)
						#gold_label.append('others')
					#gold_label.append(gold_label_list)
					#print(tmp['label_annotated'])
					#sys.exit()
		self.gold_id=gold_id
		#prediction=[]
		#for line in open('result_bug_vs_nonbug.txt'):
		#	line=line.strip().split('\t')
		#	prediction.append(float(line[-1]))
		#self.URL_maxF1_eval(prediction, gold_label)
		#sys.exit()
		#with open('result_automatic_vs_annotation.txt','a+') as f:
		#	for i in range(len(automatic_label)):
		#		f.writelines(automatic_label[i]+'\t'+gold_label[i]+'\n')
		#self.F1_Evaluation_Overall(automatic_label, gold_label)
		#sys.exit()
		#print(gold_label)
		#sys.exit()
		for file in list_of_files:
			print(file)
			with open(file, 'r') as json_file:
				data=[]
				label=[]
				for line in json_file:
					tmp = json.loads(line)
					if tmp['type'] == 'IssuesEvent':
						try:
							if tmp['payload']['issue']['labels']:
								label_flag=True
								for element in tmp['payload']['issue']['labels']:
									label_name=element['name'].lower()
									if self.label_list[0]=='bug':
										if 'bug' in label_name and 'debug' not in label_name:
											label_name = 'bug'
									else:
										if 'feature' in label_name or 'enhancement' in label_name:
											label_name = 'enhancement'
									if label_name in self.label_list[0]:
										label.append(True)
										#label.append(label_name)
										content = ''
										if tmp['payload']['issue']['title']:
											content = tmp['payload']['issue']['title'].encode('utf-8') + '\t'
										if tmp['payload']['issue']['body']:
											content += tmp['payload']['issue']['body'].encode('utf-8')
										data.append(content)
										label_flag=False
										break # single label currently
									else:
										continue
								if label_flag:
									label.append(False)
									#label.append('others')
									content=''
									if tmp['payload']['issue']['title']:
										content=tmp['payload']['issue']['title'].encode('utf-8') + '\t'
									if tmp['payload']['issue']['body']:
										content+=tmp['payload']['issue']['body'].encode('utf-8')
									data.append(content)
							else:
								if 'train' in file and self.use_unlabeled_data:
									label.append(False)
									#label.append('others')
									content = ''
									if tmp['payload']['issue']['title']:
										content = tmp['payload']['issue']['title'].encode('utf-8') + '\t'
									if tmp['payload']['issue']['body']:
										content += tmp['payload']['issue']['body'].encode('utf-8')
									data.append(content)
						except:
							print file
							print line
							sys.exit()
			if 'train' in file:
				#print(label)
				train_set=(data, label)
				#print(len(data))
				#sys.exit()
			elif 'dev' in file:
				dev_set=(data, label)
				#print(len(data))
			else:
				test_set=(data, label)
				#print(len(data))
		test_set=() # test_set is not used here
		# dev_set is replaced by gold_set, which contains 201 human annotation
		dev_set=(gold_data, gold_label)
		#self.dev_set=dev_set
		print('Number of training examples: %d' %(len(train_set[0])))
		data=train_set[0]+dev_set[0]#+test_set[0]
		data=self.Feature_Extraction(data)
		train_set=(data[:len(train_set[0])],train_set[1])
		dev_set=(data[len(train_set[0]):len(train_set[0])+len(dev_set[0])], dev_set[1])
		#test_set=(data[len(train_set[0])+len(dev_set[0]):],test_set[1])
		return (train_set, dev_set, test_set)

	def F1_Evaluation_Per_Category(self, predicted, test_label):
		for label in self.label_list:
			binary_predicted=[label in item for item in predicted]
			binary_test_label=[label in item for item in test_label]
			tp = 0.0
			fp = 0.0
			fn = 0.0
			tn = 0.0
			for i, t in enumerate(test_label):
				#if predicted[i]=='bug':
				#	print(self.dev_set[0][i])
				if binary_predicted[i] == True and binary_test_label[i] == False:
					fp += 1.0
				elif binary_predicted[i] == False and binary_test_label[i] == True:
					fn += 1.0
				elif binary_predicted[i] == True and binary_test_label[i] == True:
					tp += 1.0
				else:
					tn += 1.0
			if tp > 0:
				P = tp / (tp + fp)
				R = tp / (tp + fn)
				F = 2 * P * R / (P + R)
			else:
				P = 0
				R = 0
				F = 0

			print "Class: " + label+ "\tPRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)

	def F1_Evaluation_Overall(self,predicted, test_label):
		tp = 0.0
		fp = 0.0
		fn = 0.0
		tn = 0.0
		P=0
		R=0
		F=0
		gold_bug_enhance=0
		for i in range(len(test_label)):
			if test_label[i]!='others':
				gold_bug_enhance+=1
		#print(gold_bug_enhance)
		for i in range(len(test_label)):
			if predicted[i] in test_label[i] and predicted[i]!='others':
				tp+=1
			elif predicted[i] not in test_label[i] and predicted[i]!='others':
				fp+=1
		fn=gold_bug_enhance-tp
		if (tp + fp)> 0:
			P = tp / (tp + fp)
		if (tp + fn) > 0:
			R = tp / (tp + fn)
		if (P + R) > 0:
			F = 2 * P * R / (P + R)


		print "\tPRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)

	def Accuracy_Evaluation(self,predict_result, test_data_label):
		num_correct = 0
		for i, t in enumerate(predict_result):
			if t == test_data_label[i]:
				#print(test_data_label[i])
				num_correct += 1
		print('Overall accuracy: %.6f' % (float(num_correct) / len(test_data_label)))

	def Error_Analysis(self,predict_result, test_data_label):
		for i, t in enumerate(test_data_label):
			if test_data_label[i]!=predict_result[i] and test_data_label[i]!='others':
				print(self.data[self.test_split+i])
				print('true label: ' + test_data_label[i] +'; predicted label: '+predict_result[i])
				print('-'*24)

	def Check_Top_Weighted_Features(self, classifier, label, top_k):
		coefs = classifier.coef_[self.label_list.index(label)]
		top_rank = np.argpartition(coefs, -top_k)[-top_k:]
		my_dict = {}
		for index in top_rank:
			my_dict[self.feature_names[index]] = coefs[index]
		for key, value in sorted(my_dict.iteritems(), reverse=True, key=lambda (k, v): (v, k)):
			print key, value

	def maxF1_eval(self, predict_result, test_data_label):
		#test_data_label = ['enhancement' in item for item in test_data_label]
		counter = 0
		tp = 0.0
		fp = 0.0
		fn = 0.0
		tn = 0.0

		for i, t in enumerate(predict_result):

			if t > 0.5:
				guess = True
			else:
				guess = False
			label = test_data_label[i]
			# print guess, label
			if guess == True and label == False:
				fp += 1.0
			elif guess == False and label == True:
				fn += 1.0
			elif guess == True and label == True:
				tp += 1.0
			elif guess == False and label == False:
				tn += 1.0
			if label == guess:
				counter += 1.0
			# else:
			# print label+'--'*20
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

		try:
			P = tp / (tp + fp)
			R = tp / (tp + fn)
			F = 2 * P * R / (P + R)
		except:
			P = 0
			R = 0
			F = 0

		print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
		print "ACCURACY: %s" % (counter / len(predict_result))

		# print "# true pos:", tp
		# print "# false pos:", fp
		# print "# false neg:", fn
		# print "# true neg:", tn
		maxF1 = 0
		P_maxF1 = 0
		R_maxF1 = 0
		p_list = []
		r_list = []
		probs = predict_result
		sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
		sortedindex.reverse()

		truepos = 0
		falsepos = 0
		for sortedi in sortedindex:
			if test_data_label[sortedi] == True:
				truepos += 1
			elif test_data_label[sortedi] == False:
				falsepos += 1
			precision = 0
			if truepos + falsepos > 0:
				precision = truepos / (truepos + falsepos)
			#print(precision)
			recall = truepos / (tp + fn)
			#print(recall)
			#print precision, recall
			f1 = 0
			if precision + recall > 0:
				f1 = 2 * precision * recall / (precision + recall)
				p_list.append(precision)
				r_list.append(recall)
				if f1 > maxF1:
					# print probs[sortedi]
					maxF1 = f1
					P_maxF1 = precision
					R_maxF1 = recall
		print "PRECISION: %s, RECALL: %s, max_F1: %s" % (P_maxF1, R_maxF1, maxF1)
		plt.ylabel('Precision', fontsize=24, fontweight='medium')
		plt.xlabel('Recall', fontsize=24, fontweight='medium')
		plt.tick_params(labelsize=18)

		plt.ylim((0.0, 1.0))
		plt.xlim((0.0, 1.0))
		plt.plot(r_list, p_list, linewidth=2.0)
		plt.legend(numpoints=1, loc=0, scatterpoints=1, frameon=False, fontsize=20)
		plt.show()
		return maxF1

	def Logistic_Regression(self, train_set, dev_set, test_set):
		train_data=train_set[0]
		train_data_label=train_set[1]
		test_data=dev_set[0]
		test_data_label=dev_set[1]

		if self.saved_model:
			if os.path.isfile(self.saved_model):
				classifier=pickle.load(open(self.saved_model, 'rb'))
			else:
				logistic = linear_model.LogisticRegression()
				classifier = logistic.fit(train_data, train_data_label)
				pickle.dump(classifier, open(self.saved_model, 'wb'))
		else:
			print('LR')
			logistic = linear_model.LogisticRegression()
			classifier = logistic.fit(train_data, train_data_label)
		#self.Check_Top_Weighted_Features(classifier, 'bug', 20)
		#predict_result = classifier.predict(test_data)
		probas = classifier.predict_proba(test_data)
		#for item in probas:
		#	print(item)
		probas=probas[:,1]
		print('-' * 40)
		print('Number of testing examples: %d' %(len(test_data)))
		#print(metrics.confusion_matrix(y_pred=predict_result, y_true=test_data_label))
		#self.Accuracy_Evaluation(predict_result, test_data_label)
		#self.F1_Evaluation_Per_Category(predict_result, test_data_label)
		#self.F1_Evaluation_Overall(predict_result, test_data_label)
		self.maxF1_eval(probas, test_data_label)
		#with open('result_bug_vs_non-bug.txt','w') as f:
		#	for i in range(len(predict_result)):
		#		f.writelines(str(predict_result[i])+'\t'+str(test_data_label[i])+'\t'+self.gold_id[i]+'\t'+str(probas[i])+'\n')
		#print('-'*40)
		#self.Error_Analysis(predict_result,test_data_label)

	def Naive_Bayes(self, train_set, dev_set):
		train_data = train_set[0]
		train_data_label = train_set[1]
		test_data = dev_set[0]
		test_data_label = dev_set[1]

		clf = GaussianNB()
		clf.fit(train_data, train_data_label)
		predict_result = clf.predict(test_data)
		print('-' * 40)
		print(len(test_data))
		self.Accuracy_Evaluation(predict_result, test_data_label)
		self.F1_Evaluation(predict_result, test_data_label)

	def SVM(self, train_set, dev_set):
		train_data = train_set[0]
		train_data_label = train_set[1]
		test_data = dev_set[0]
		test_data_label = dev_set[1]

		clf = SVC()
		clf.fit(train_data, train_data_label)
		predict_result = clf.predict(test_data)
		print('-' * 40)
		print(len(test_data))
		self.Accuracy_Evaluation(predict_result, test_data_label)
		self.F1_Evaluation(predict_result, test_data_label)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--tag', type=str, default='bug',
	                    help='Currently supported tags: bug and enhancement.')
	parser.add_argument('--use_unlabeled_data', type=str, default='false',
	                    help='True or False')
	parser.add_argument('--use_saved_model', type=str, default='true',
	                    help='True or False')
	args = parser.parse_args()
	if args.tag=='bug':
		labels=['bug','others']
	elif args.tag=='enhancement':
		labels=['enhancement','others']
	else:
		print('wrong input!')
		sys.exit()
	if args.use_unlabeled_data=='true' or args.use_unlabeled_data=='True':
		use_unlabeled_data=True
	else:
		use_unlabeled_data=False
	if args.use_saved_model=='true' or args.use_saved_model=='True':
		if 'bug' in args.tag:
			saved_model='lr_model_bug.sav'
		else:
			saved_model = 'lr_model_enhancement.sav'
	else:
		saved_model=''
	my_classifier=Github_Issue_Classifier(data_folder='data/',label_list=labels, use_unlabeled_data=use_unlabeled_data, saved_model=saved_model)
	#train_set, dev_set, test_set = my_classifier.Data_Preprocess_gz_file()()
	train_set, dev_set, test_set=my_classifier.Data_Preprocess_txt_file()
	my_classifier.Logistic_Regression(train_set, dev_set, test_set)
