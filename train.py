from torch.utils.data import Dataset,DataLoader
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
import torch.nn as nn
#准备数据集

data_base_path = r'F:\DownloadFile\aclImdb'

def tokenlize(content):
	content = re.sub('<.*?>',' ',content)
	fileters = [':','\t','\n','\x97','\x96','#','$','%','&','\.']
	content = re.sub('|'.join(fileters),' ',content)
	tokens = [i.strip().lower() for i in content.split()]
	return tokens

class Word2Sequence():
	UNK_TAG = 'UNK'
	PAD_TAG = 'PAD'

	UNK = 0
	PAD = 1
	def __init__(self):
		self.dict = {
		self.UNK_TAG:self.UNK,
		self.PAD_TAG:self.PAD
		}
		self.count = {}
	def fit(self,sentence):
		for word in sentence:
			self.count[word] = self.count.get(word,0) + 1
	def build_vocab(self,min = 5,max = None,max_features = None):
		if min is not None:
			self.count = {word:value for word,value in self.count.items() if value > min}
		if max is not None:
			self.count = {word:value for word,value in self.count.items() if value < max}
		if max_features is not None:
			temp = sorted(self.count.items(),key = lambda x:x[-1],reverse = True)[:max_features]
			self.count = dict(temp)
		for word in self.count:
			self.dict[word] = len(self.dict)
		self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))
	def transform(self,sentence,max_len = None):
		if max_len is not None:
			if max_len > len(sentence):
				sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
			if max_len < len(sentence):
				sentence = sentence[:max_len]
		return [self.dict.get(word,self.UNK) for word in sentence]
	def inverse_transform(self,indices):
		return [self.inverse_dict.get(idx) for idx in indices]
	def __len__(self):
		return len(self.dict)

class IMDBDataset(Dataset):
	def __init__(self,train = True):
		self.train_data_path = r'F:\DownloadFile\aclImdb\train'
		self.test_data_path = r'F:\DownloadFile\aclImdb\test'
		data_path = self.train_data_path if train else self.test_data_path
		#把所以的文件名放入列表
		temp_data_path = [os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
		self.total_file_path = []#所有的品论文件
		for path in temp_data_path:
			file_name_list = os.listdir(path)
			file_path_list = [os.path.join(path,i) for i in file_name_list if i.endswith('.txt')]
			self.total_file_path.extend(file_path_list)
	def __getitem__(self,index):
		file_path = self.total_file_path[index]
		#获取标签
		label_str = file_path.split('\\')[-2]
		label = 0 if label_str == 'neg' else 1
		#获取内容
		tokens = tokenlize(open(file_path,encoding = 'utf-8').read())
		return tokens,label
	def __len__(self):
		return len(self.total_file_path)


ws = pickle.load(open(r'F:\DownloadFile\aclImdb\train\ws.pkl','rb'))

def collate_fn(batch):
	content,label = list(zip(*batch))
	content = [ws.transform(i,max_len = 200) for i in content]
	content = torch.LongTensor(content)
	label = torch.LongTensor(label)
	return content,label

def get_dataloader(train = True):
	imda_dataset = IMDBDataset(train)
	data_loader = DataLoader(imda_dataset,batch_size = 128,shuffle = True,collate_fn = collate_fn)
	return data_loader


class MyModel(nn.Module):
	def __init__(self):
		super(MyModel,self).__init__()
		self.embedding = nn.Embedding(len(ws),100)
		self.lstm1 = nn.LSTM(input_size = 100,hidden_size = 128,num_layers = 2,batch_first = True,bidirectional = True,dropout = 0.4)
		self.lstm2 = nn.LSTM(input_size = 256,hidden_size = 128,num_layers = 1,batch_first = True,bidirectional = False,dropout = 0.4)
		self.fc1 = nn.Linear(128,54)
		self.fc2 = nn.Linear(54,2)
	def forward(self,input):
		out = self.embedding(input)
		#print(out.shape)
		out,(hn,cn) = self.lstm1(out)
		#print(out.shape)
		out,(hn,cn) = self.lstm2(out)
		#print(out.shape)
		output_f = hn[-1,:,:]
		#print(output_f.shape)
		#output_b = hn[-1,:,:]
		#output = torch.cat([output_f,output_b],dim = -1)
		output = self.fc1(output_f)
		#print(output.shape)
		output = F.relu(output)
		#print(output.shape)
		output = self.fc2(output)
		#print(output.shape)
		return F.log_softmax(output,dim = -1)

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(),0.001)
def train(epoch):
	train_data_loader = get_dataloader(train = True)
	for idx,(input,target) in tqdm(enumerate(train_data_loader),total = len(train_data_loader)):
		output = model(input)
		optimizer.zero_grad()
		loss = F.nll_loss(output,target)
		loss.backward()
		optimizer.step()
		print(loss.item())

		if idx % 100 == 0:
			torch.save(model.state_dict(),r'F:\DownloadFile\aclImdb\train\train.pkl')
			torch.save(optimizer.state_dict(),r'F:\DownloadFile\aclImdb\train\optimizer.pkl')


def myeval():
	loss_list = []
	acc_list = []
	test_data_loader = get_dataloader(train = False)
	for idx,(input,target) in tqdm(enumerate(test_data_loader),total = len(test_data_loader)):
		with torch.no_grad():
			output = model(input)
			cur_loss = F.nll_loss(output,target)
			loss_list.append(cur_loss.item())
			pred = output.max(dim = -1)[-1]
			cur_acc = pred.eq(target).float().mean()
			acc_list.append(cur_acc.item())
	print('eval loss: acc:',np.mean(loss_list),np.mean(acc_list))

info = input()
if info == 'train':
	print('train')
	for i in range(30):
		train(i)
elif info == 'test':
	print('test')
	model.load_state_dict(torch.load(r'F:\DownloadFile\aclImdb\train\train.pkl'))
	myeval()
else:
	print('false info')


# for i in range(10):
# 	train(i)
# model.load_state_dict(torch.load(r'F:\DownloadFile\aclImdb\train\train.pkl'))
# myeval()

# path = r'F:\DownloadFile\aclImdb\train'
# ws = Word2Sequence()
# temp_path = [os.path.join(path,'pos'),os.path.join(path,'neg')]
# for data_path in temp_path:
# 	file_paths = [os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')]
# 	for file_path in tqdm(file_paths):
# 		sentence = tokenlize(open(file_path,encoding = 'utf-8').read())
# 		ws.fit(sentence)
# ws.build_vocab(min = 10,max_features = 10000)
# pickle.dump(ws,open(r'F:\DownloadFile\aclImdb\train\ws.pkl','wb'))
# print(len(ws))