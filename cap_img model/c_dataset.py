import numpy as np
import torch
from transformers import *
from torch.utils.data import Dataset
import _pickle as cPickle
import os
import json
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm
import utils
import random


def _create_entry(question, answer, caption, attr_cls):
    if None!=answer:
        
        #answerからimage_idとquestion_idを取り除く
        answer.pop('image_id') #answerから'image_id'を取り除きそれを返す
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer,
        'caption' : caption,
        'attr_cls' : attr_cls
        }
    return entry


def _load_dataset(dataroot, name, label2ans):
    
    
    #question_path = os.path.join(
        #dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' %(name + '2014' if 'test'!=name[:4] else name))
    #question_path = os.path.join(
        #dataroot, 'vqacp/VQA-CP data/vqa/vqacp2/raw/annotations/', 'vqacp_v2_%s_questions.json' %name)
    #questions = sorted(json.load(open(question_path)),
                       #key=lambda x: x['question_id'])
    question_path = os.path.join(dataroot, '%s_questions.pkl' %(name))
    questions = cPickle.load(open(question_path, 'rb'))
    questions = sorted(questions, key=lambda x: x['question_id'])
    print('the number of questions', len(questions))
    
    #train (for faster rcnn features)
    attr_cls_train_path = os.path.join(dataroot, 'cp_train.pkl')
    cls_train = cPickle.load(open(attr_cls_train_path, 'rb'))
    cls_train = sorted(cls_train, key=lambda x: x['image_id'])
    #val
    attr_cls_val_path = os.path.join(dataroot, 'cp_val.pkl')
    cls_val = cPickle.load(open(attr_cls_val_path, 'rb'))
    cls_val = sorted(cls_val, key=lambda x: x['image_id'])
    
    if 'test'!=name[:4]:
        
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        
        entries = []
        
        #train (for faster rcnn features)
        ins_path_train = 'data/annotations/instances_train2014.json'
        coco_train = COCO(ins_path_train)
        cats_train = coco_train.loadCats(coco_train.getCatIds())
        catIds_train = coco_train.getCatIds(catNms=cats_train)
        imgIds_train = coco_train.getImgIds(catIds=catIds_train)
        imgIds_train = sorted(imgIds_train) #sort imgIds in order of image id
        #print('the number of images: ', len(imgIds))
        # initialize COCO api for caption annotations
        annFile_train = '{}/annotations/captions_train2014.json'.format(dataroot)
        coco_caps_train=COCO(annFile_train)
        
        #val 
        ins_path_val = 'data/annotations/instances_val2014.json'
        coco_val = COCO(ins_path_val)
        cats_val = coco_val.loadCats(coco_val.getCatIds())
        catIds_val = coco_val.getCatIds(catNms=cats_val)
        imgIds_val = coco_val.getImgIds(catIds=catIds_val)
        imgIds_val = sorted(imgIds_val) #sort imgIds in order of image id
        #print('the number of images: ', len(imgIds))
        # initialize COCO api for caption annotations
        annFile_val = '{}/annotations/captions_val2014.json'.format(dataroot)
        coco_caps_val = COCO(annFile_val)
        
        
        #for sentence embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('bert-base-nli-mean-tokens')


        j = 0 #for counting
        for question, answer in tqdm(zip(questions, answers)):
            img_id = question['image_id']
            
            #faster rcnn features
            if question['coco_split'] == 'train2014':
                cls = cls_train
            else:
                cls = cls_val
            attr_cls = [c['attr_cls'] for c in cls if c['image_id'] == img_id]
            attr_cls = attr_cls[0]
            
            
            #captions 
            coco_name = question['coco_split']
            if coco_name == 'train2014':
                #print('use train')
                coco = coco_train
                imgIds = imgIds_train
                coco_caps = coco_caps_train
            else:
                #print('use val')
                coco = coco_val
                imgIds = imgIds_val
                coco_caps = coco_caps_val
            imgIds = coco.getImgIds(imgIds = img_id)
            img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            capIds = coco_caps.getAnnIds(imgIds=img['id']) #get annotation ids that satisfy given filter conditions
            captions = coco_caps.loadAnns(capIds)
            #print(captions)
            
            #random selection
            #i = random.randint(0, 4)
            #caption = captions[i]['caption']
            
            #select max cos_sim
            sentences = [question['question']]
            for i in range(5):
                sentences.append(captions[i]['caption'])
            
            sentence_embeddings = model.encode(sentences)
            maxsim = 0
            maxid = 1
            for i in range(5):
                cosim = cos_sim(sentence_embeddings[0], sentence_embeddings[i+1])
                if cnt < 5:
                        print('cosim and cap')
                        print(cosim)
                        print(sentences[i+1])
                if cosim > maxsim:
                    maxsim = cosim
                    maxid = i + 1
            caption = sentences[maxid]
            
            
            entry = _create_entry(question, answer, caption, attr_cls)
            entries.append(entry)
            
            #for debag
            if j % 100000 == 0:
                print(entry)
                
            j += 1
            
    else: # test2015 #have not finished
        entries = []
        for question in questions:
            img_id = question['image_id']
            #if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                #entries.append(_create_entry(img_id2val[img_id], question, None)) #各質問のエントリを作る(q_id, im_id, im, q, ans)
            entries.append(question, caption, None)
            
    print('entries done')
    print(entries[0])
    
    entry_file = os.path.join('data', 'c_entry_%s.pkl' % name) 
    cPickle.dump(entries, open(entry_file, 'wb'))

    
    
    return entries





class VQABERTDataset(Dataset):
    def __init__(self, name, use_ans, dataroot='data'):
        super(VQABERTDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test']
        
        self.use_ans = use_ans
        
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        entry_path = 'data/c_entry_%s.pkl' %name
        if os.path.isfile(entry_path):
            print('find the entries')
            self.entries = cPickle.load(open(entry_path, 'rb'))
            print(self.entries[1])
        else:
            print('make the entries')
            self.entries = _load_dataset(dataroot, name, self.label2ans)
            
        print(len(self.entries))
         
        tokenized_entry_path = os.path.join('data', 'c_tokenized_entry_%s.pkl' % name) 
        if os.path.isfile(tokenized_entry_path):
            print('find the tokenized entries')
            self.entries = cPickle.load(open(tokenized_entry_path, 'rb'))
        else:
            self.tokenize(name) #エントリ（全て）にトークン化された質問を追加
            
        self.tensorize()


    def tokenize(self, name, max_length=128):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        print('start to tokenize')
        for entry in tqdm(self.entries):
            #caption
            c_encoded_dict = self.tokenizer.encode_plus(entry['caption'], entry['question'], add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt') #質問とキャプションを前処理
                
            entry['c_encoded_dict'] = c_encoded_dict #エントリにトencoded_dictを追加
            
            #faster rcnn
            for i, at_c in enumerate(entry['attr_cls']):
                if i == 0:
                    attr_cls = at_c
                else:
                    attr_cls = ','.join([attr_cls, at_c])
                
            v_encoded_dict = self.tokenizer.encode_plus(attr_cls, entry['question'], add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
            
            entry['v_encoded_dict'] = v_encoded_dict
            
            
        #tokenized_entry_path = os.path.join('data', 'c_tokenized_entry_%s.pkl' % name) 
        #cPickle.dump(self.entries, open(tokenized_entry_path, 'wb'))
        
        
    def tensorize(self):

        for entry in self.entries:
            #####turn q into tensor! (or just string is fine)
            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    #entryの'answer'に'labels'と'scores'がある
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None


    def __getitem__(self, index):
        entry = self.entries[index] #indexの質問のエントリentry = {'question_id' : question['question_id'],'image_id': question['image_id'],'image': img,'question': question['question'],'answer':answer}

        c_encoded_dict = entry['c_encoded_dict'] #トークン化された質問文
        v_encoded_dict = entry['v_encoded_dict']
        
        question_id = entry['question_id']
        question = entry['question']
        
        c_input_ids = c_encoded_dict['input_ids']
        c_input_mask = c_encoded_dict['attention_mask']
        c_input_token_type = c_encoded_dict['token_type_ids']
        
        v_input_ids = v_encoded_dict['input_ids']
        v_input_mask = v_encoded_dict['attention_mask']
        v_input_token_type = v_encoded_dict['token_type_ids']
        
        caption = entry['caption']
        visual_concepts = entry['attr_cls']
        
        answer = entry['answer']

        if None!=answer:
            labels = answer['labels'] #answerにはlabelとscoreがある(両方リスト)
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates) #ansの候補の数分の0を要素にもつtensorを生成
            if labels is not None:
                target.scatter_(0, labels, scores) #candidate answerのうち, labelsのインデックスのところを,対応するscoresの値にして, その他を0にする
            return c_input_ids, c_input_mask, c_input_token_type, v_input_ids, v_input_mask, v_input_token_type, target, question, question_id, caption, visual_concepts
        else:
            return c_input_ids, c_input_mask, c_input_token_type, v_input_ids, v_input_mask, v_input_token_type, question_id


    def __len__(self):
        return len(self.entries)
