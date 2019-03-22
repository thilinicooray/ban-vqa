import torch
import random
from collections import OrderedDict
import csv
import nltk
import json

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set, role_questions, dataset_folder):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = ['#UNK#']
        label_frequency = {}
        self.max_role_count = 0
        self.question_words = {}
        self.max_q_word_count = 0
        self.verb_question = {}
        self.q_idx = json.load(open(dataset_folder +'/commonq.json'))

        for img_id, question in role_questions.items():
             question = "what is the action happening?"
             self.verb_question[img_id] = question
             words = nltk.word_tokenize(question)
             words = words[:-1] #ignore ? mark
             if len(words) > self.max_q_word_count:
                 self.max_q_word_count = len(words)
             #print('q words :', words)

             for word in words:
                 if word not in self.question_words:
                     self.question_words[word] = len(self.question_words)

        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)
                self.verb2_role_dict[current_verb] = []

            for frame in img['frames']:
                for role,label in frame.items():
                    if role not in self.role_list:
                        self.role_list.append(role)
                    if role not in self.verb2_role_dict[current_verb]:
                        self.verb2_role_dict[current_verb].append(role)
                    if len(self.verb2_role_dict[current_verb]) > self.max_role_count:
                        self.max_role_count = len(self.verb2_role_dict[current_verb])
                    if label not in self.label_list:
                        if label not in label_frequency:
                            label_frequency[label] = 1
                        else:
                            label_frequency[label] += 1
                        #only labels occur at least 20 times are considered
                        if label_frequency[label] == 20:
                            self.label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count,
              '\n\t max q word count:', self.max_q_word_count)

        self.verb_question_words = self.question_words


        verb2role_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb = []
            for role in current_role_list:
                role_id = self.role_list.index(role)
                role_verb.append(role_id)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_verb.append(len(self.role_list))

            verb2role_list.append(torch.tensor(role_verb))

        self.verb2role_list = torch.stack(verb2role_list)
        self.verb2role_encoding = self.get_verb2role_encoding()
        '''print('verb to role list :', self.verb2role_list.size())

        print('unit test verb and roles: \n')
        verb_test = [4,57,367]
        for verb_id in verb_test:
            print('verb :', self.verb_list[verb_id])

            role_list = self.verb2role_list[verb_id]

            for role in role_list:
                if role != len(self.role_list):
                    print('role : ', self.role_list[role])'''

    def encode(self, item, id):
        verb = self.verb_list.index(item['verb'])
        verb_q = self.get_verbq(id)
        roles = self.get_role_ids(verb)
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, verb_q, roles, labels

    def encode_verb(self, item):
        verb = self.verb_list.index(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, labels

    def get_verb2role_encoding(self):
        verb2role_embedding_list = []

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(1)


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(0)

            verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

        return verb2role_embedding_list

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_role_ids_batch(self, verbs):
        role_batch_list = []
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_role_questions_batch(self, verbs):
        role_batch_list = []
        q_len_batch = []

        for verb_id in verbs:
            rquestion_tokens = []
            q_len = []
            verb = self.verb_list[verb_id]
            current_role_list = self.verb2_role_dict[verb]

            for role in current_role_list:
                question = self.vrole_question[verb+'_'+role]
                #print('question :', question)
                q_tokens = []
                words = nltk.word_tokenize(question)
                words = words[:-1]
                for word in words:
                    q_tokens.append(self.question_words[word])
                padding_words = self.max_q_word_count - len(q_tokens)

                for w in range(padding_words):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(len(words))

            role_padding_count = self.max_role_count - len(current_role_list)

            #todo : how to handle below sequence making for non roles properly?
            for i in range(role_padding_count):
                q_tokens = []
                for k in range(0,self.max_q_word_count):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(0)
            role_batch_list.append(torch.stack(rquestion_tokens,0))
            q_len_batch.append(torch.tensor(q_len))

        return torch.stack(role_batch_list,0), torch.stack(q_len_batch,0)

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

    def get_role_questions(self, verb):
        rquestion_tokens = []
        q_len = []
        current_role_list = self.verb2_role_dict[verb]

        for role in current_role_list:
            question = self.vrole_question[verb+'_'+role]
            #print('question :', question)
            q_tokens = []
            words = nltk.word_tokenize(question)
            words = words[:-1]
            for word in words:
                q_tokens.append(self.question_words[word])
            padding_words = self.max_q_word_count - len(q_tokens)

            for w in range(padding_words):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(len(words))

        role_padding_count = self.max_role_count - len(current_role_list)

        #todo : how to handle below sequence making for non roles properly?
        for i in range(role_padding_count):
            q_tokens = []
            for k in range(0,self.max_q_word_count):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(0)

        return torch.stack(rquestion_tokens,0), torch.tensor(q_len)


    def get_verbq(self, id):
        vquestion_tokens = []
        question = self.verb_question[id]

        words = nltk.word_tokenize(question)
        words = words[:-1]
        for word in words:
            vquestion_tokens.append(self.question_words[word])
        padding_words = self.max_q_word_count - len(vquestion_tokens)

        for w in range(padding_words):
            vquestion_tokens.append(len(self.question_words))

        return torch.tensor(vquestion_tokens)

    def get_common_verbq(self, batch_size):
        all_q_idx = []
        for i in range(batch_size):
            q_idx = []
            question = 'what is the action happening'

            words = question.split()
            for word in words:
                q_idx.append(self.question_words[word])
            padding_words = self.max_q_word_count - len(q_idx)

            for w in range(padding_words):
                q_idx.append(len(self.question_words))

            all_q_idx.append(torch.tensor(q_idx))

        return torch.stack(all_q_idx,0)

    def get_common_verbq_encoding(self, batch_size):
        all_q_idx = []

        for i in range(batch_size):

            all_q_idx.append(torch.tensor(self.q_idx))

        return torch.stack(all_q_idx,0)

    def get_label_ids(self, frames):
        all_frame_id_list = []
        for frame in frames:
            label_id_list = []
            for role,label in frame.items():
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(self.get_num_labels())

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels

    def get_adj_matrix(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_adj_matrix_noself(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx1 in range(0,role_count):
                adj[idx1][idx1] = 0
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def getadj(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            '''encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose'''
            adj = torch.zeros(6, 6)
            for idx in range(0,6):
                adj[idx][idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_mask(self, verb_ids, org_tensor):
        org = org_tensor.clone()
        org_reshaped = org.view(len(verb_ids), self.max_role_count, -1, org.size(2))
        for i in range(0, len(verb_ids)):
            role_encoding = self.verb2role_encoding[verb_ids[i]]
            for j in range(0, len(role_encoding)):
                #print('i:j', i,j)
                if role_encoding[j] == 0:
                    org_reshaped[i][j:] = 0
                    break
        return org_reshaped.view(org_tensor.size())

    def get_extended_encoding(self, verb_ids, dim):
        encoding_list = []
        for id in verb_ids:
            encoding = self.verb2role_encoding[id]

            encoding = torch.unsqueeze(torch.tensor(encoding),1)
            #print('encoding unsqe :', encoding.size())
            encoding = encoding.repeat(1,dim)
            #encoding = torch.squeeze(encoding)
            #print('extended :', encoding.size(), encoding)
            encoding_list.append(encoding)

        return torch.stack(encoding_list).type(torch.FloatTensor)
