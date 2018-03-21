from __future__ import division
import cPickle as pickle
import argparse
import torch
import numpy as np
import time
from datetime import timedelta
from torch.autograd import Variable
from model import CNN_Text
import torch.nn.functional as F


def maxF1_eval(predict_result, test_data_label):
	# test_data_label = ['enhancement' in item for item in test_data_label]
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

	# print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	# print "ACCURACY: %s" % (counter / len(predict_result))

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
		# print(precision)
		recall = truepos / (tp + fn)
		# print(recall)
		# print precision, recall
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

def create_batch(train_data,from_index, train_labels):
	to_index=from_index+args.batch_size
	if to_index>len(train_data):
		to_index=len(train_data)
	max_len=0
	for i in range(from_index, to_index):
		if len(train_data[i])>max_len:
			max_len=len(train_data[i])
	max_len+=2
	lsent = train_data[from_index]
	lsent = ['bos']+lsent + ['oov' for k in range(max_len -1 - len(lsent))]
	#print(lsent)
	left_sents = [[word2id[word] for word in lsent]]

	#print(train_labels[from_index])
	if train_labels[from_index]==True:
		labels = [1]
	else:
		labels = [0]

	for i in range(from_index+1, to_index):

		lsent=train_data[i]
		lsent=['bos']+lsent+['oov' for k in range(max_len -1 - len(lsent))]
		#print(lsent)
		left_sents.append([word2id[word] for word in lsent])

		if train_labels[i] == True:
			labels.append(1)
		else:
			labels.append(0)

	left_sents=Variable(torch.LongTensor(left_sents))
	labels=Variable(torch.LongTensor(labels))

	if torch.cuda.is_available():
		left_sents=left_sents.cuda()
		labels=labels.cuda()
	return left_sents, labels

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CNN text classificer')
	# learning
	parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
	parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 256]')
	parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 64]')
	parser.add_argument('-log-interval', type=int, default=1,
	                    help='how many steps to wait before logging training status [default: 1]')
	parser.add_argument('-test-interval', type=int, default=100,
	                    help='how many steps to wait before testing [default: 100]')
	parser.add_argument('-save-interval', type=int, default=500,
	                    help='how many steps to wait before saving [default:500]')
	parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
	parser.add_argument('-early-stop', type=int, default=1000,
	                    help='iteration numbers to stop without performance increasing')
	parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
	# data
	parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
	# model
	parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
	parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
	parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
	parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
	parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
	                    help='comma-separated kernel size to use for convolution')
	parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
	# device
	parser.add_argument('-device', type=int, default=-1,
	                    help='device to use for iterate data, -1 mean cpu [default: -1]')
	parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
	# option
	parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
	parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
	parser.add_argument('-test', action='store_true', default=False, help='train or test')
	args = parser.parse_args()
	#load data
	print('load data...')
	task='bug'
	if task=='enhancement':
		train_data=pickle.load(open('data/train_merged_enhancement.toks', "rb"))
		train_labels=pickle.load(open('data/train_merged_enhancement.labels', "rb"))
		#train_data = pickle.load(open('data/train_enhancement.toks', "rb"))
		#train_labels = pickle.load(open('data/train_enhancement.labels', "rb"))
		test_data=pickle.load(open('data/test_enhancement.toks', "rb"))
		test_labels=pickle.load(open('data/test_enhancement.labels', "rb"))
	else:
		train_data = pickle.load(open('data/train_merged_bug.toks', "rb"))
		train_labels = pickle.load(open('data/train_merged_bug.labels', "rb"))
		#train_data = pickle.load(open('data/train_bug.toks', "rb"))
		#train_labels = pickle.load(open('data/train_bug.labels', "rb"))
		test_data = pickle.load(open('data/test_bug.toks', "rb"))
		test_labels = pickle.load(open('data/test_bug.labels', "rb"))
	#print(train_data[123])
	#print(train_labels[123])
	print(len(train_data))
	print(len(test_data))
	print('task: '+task)
	vocab = set()
	for pair in train_data:
		left = pair
		vocab |= set(left)
	for pair in test_data:
		left = pair
		vocab |= set(left)
	tokens = list(vocab)
	tokens.append('oov')
	tokens.append('bos')
	id = 0
	word2id={}
	for word in tokens:
		word2id[word] = id
		id+=1
	args.embed_num = len(tokens)
	args.class_num = 2
	args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

	#print("\nParameters:")
	#for attr, value in sorted(args.__dict__.items()):
	#	print("\t{}={}".format(attr.upper(), value))

	model = CNN_Text(args)

	if torch.cuda.is_available():
		model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	report_interval=5000
	for epoch in range(1, args.epochs + 1):
		train_batch_i = 0
		batch_counter=0
		accumulated_loss=0
		train_sents_scaned=0
		train_num_correct=0
		model.train()
		print('--' * 20)
		start_time = time.time()
		train_data = np.array(train_data)
		train_labels=np.array(train_labels)
		rand_idx = np.random.permutation(len(train_data))
		train_data = train_data[rand_idx]
		train_labels = train_labels[rand_idx]
		while train_batch_i < len(train_data):
			sent_batch, label_batch=create_batch(train_data, train_batch_i, train_labels)
			train_batch_i += len(label_batch)
			train_sents_scaned+=len(label_batch)
			optimizer.zero_grad()
			logit = model(sent_batch)
			result = logit.data.cpu().numpy()
			a = np.argmax(result, axis=1)
			b = label_batch.data.cpu().numpy()
			train_num_correct += np.sum(a == b)
			loss = F.cross_entropy(logit, label_batch)
			loss.backward()
			optimizer.step()
			accumulated_loss += loss.data[0]
			batch_counter+=1
			if batch_counter % report_interval == 0:
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t train batch loss: %f' % (accumulated_loss / train_sents_scaned)
				msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
				print(msg)
				# valid after each epoch
				model.eval()
				test_batch_i=0
				test_num_correct=0
				#batch_counter=0
				accumulated_loss=0
				pred=[]
				gold=[]
				while test_batch_i < len(test_data):
					sent_batch, label_batch = create_batch(test_data, test_batch_i, test_labels)
					test_batch_i += len(label_batch)
					logit = model(sent_batch)
					result = F.softmax(logit).data.cpu().numpy()
					pred.extend(result[:,1])
					a = np.argmax(result, axis=1)
					b = label_batch.data.cpu().numpy()
					gold.extend(b)
					#print(a)
					#print(b)
					test_num_correct += np.sum(a == b)
					batch_counter+=1
					loss = F.cross_entropy(logit, label_batch)
					loss.backward()
					accumulated_loss += loss.data[0]
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t test batch loss: %f' % (accumulated_loss / len(test_data))
				msg += '\t test accuracy: %f' % (test_num_correct / len(test_data))
				print(msg)
				maxF1_eval(pred, gold)
				print('\n')
				model.train()
		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))