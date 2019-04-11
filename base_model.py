"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import torchvision as tv
import utils
import utils_imsitu
from attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter

import torchvision as tv

class resnet_152_features(nn.Module):
    def __init__(self):
        super(resnet_152_features, self).__init__()
        self.resnet = tv.models.resnet152(pretrained=True)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x


class BanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        return logits, att

class BanModelGrid(nn.Module):
    def __init__(self, conv_net, dataset, w_emb, q_emb, v_att, b_net, q_prj, classifier, op, glimpse):
        super(BanModelGrid, self).__init__()
        self.dataset = dataset
        self.conv_net = conv_net
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.classifier = classifier
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        img_features = self.conv_net(v)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img = img_features.view(batch_size, n_channel, -1)
        v = img.permute(0, 2, 1)

        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h

            atten, _ = logits[:,g,:,:].max(2)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        logits = self.classifier(q_emb.sum(1))

        return logits, att

class BanModel_ImSitu(nn.Module):
    def __init__(self, dataset, conv_net, w_emb, q_emb, v_att, b_net, q_prj, classifier, op, glimpse):
        super(BanModel_ImSitu, self).__init__()
        self.op = op
        self.glimpse = glimpse
        self.dataset = dataset
        self.conv_net = conv_net
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.classifier = classifier
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()


    def forward(self, img, q):
        '''
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        '''

        #get cnn feat from images
        img_features = self.conv_net(img)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img = img_features.view(batch_size, n_channel, -1)
        img = img.permute(0, 2, 1)

        img = img.expand(self.dataset.encoder.max_role_count,img.size(0), img.size(1), img.size(2))
        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size* self.dataset.encoder.max_role_count, -1, n_channel)

        q = q.view(batch_size* self.dataset.encoder.max_role_count, -1)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(img, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(img, q_emb, att[:,g,:,:]) # b x l x h

            atten, _ = logits[:,g,:,:].max(2)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        logits = self.classifier(q_emb.sum(1))

        role_label_pred = logits.contiguous().view(batch_size, self.dataset.encoder.max_role_count, -1)

        return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                for j in range(0, self.dataset.encoder.max_role_count):
                    frame_loss += utils_imsitu.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.dataset.encoder.get_num_labels())
                frame_loss = frame_loss/len(self.dataset.encoder.verb2_role_dict[self.dataset.encoder.verb_list[gt_verbs[i]]])
                #print('frame loss', frame_loss, 'verb loss', verb_loss)
                loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

class BanModel_flickr(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, op, glimpse):
        super(BanModel_flickr, self).__init__()
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.alpha = torch.Tensor([1.]*(glimpse))

    # features, spatials, sentence, e_pos, target
    def forward(self, v, b, q, e, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch, seq_length]
        e: [batch, num_entities]

        return: logits, not probs
        """
        assert q.size(1) > e.data.max(), 'len(q)=%d > e_pos.max()=%d' % (q.size(1), e.data.max())
        MINUS_INFINITE = -99
        if 's' in self.op:
            v = torch.cat([v, b], 2)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        # entity positions
        q_emb = utils.batched_index_select(q_emb, 1, e)

        att = self.v_att.forward_all(v, q_emb, True, True, MINUS_INFINITE)  # b x g x v x q
        mask = (e == 0).unsqueeze(1).unsqueeze(2).expand(att.size())
        mask[:, :, :, 0].data.fill_(0)  # at least one entity per sentence
        att.data.masked_fill_(mask.data, MINUS_INFINITE)

        return None, att


def build_ban(dataset, num_hid, op='', gamma=4, task='vsrl'):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    v_att = BiAttention(2048, num_hid, num_hid, gamma)
    if task == 'vqa':
        b_net = []
        q_prj = []
        c_prj = []
        objects = 10  # minimum number of boxes
        for i in range(gamma):
            b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
        counter = Counter(objects)
        return BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma)
    elif task == 'flickr':
        return BanModel_flickr(w_emb, q_emb, v_att, op, gamma)

    elif task == 'vsrl':
        conv_net = resnet_152_features()
        b_net = []
        q_prj = []
        for i in range(gamma):
            b_net.append(BCNet(2048, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, dataset.encoder.get_num_labels(), .5)
        return BanModel_ImSitu(dataset, conv_net, w_emb, q_emb, v_att, b_net, q_prj, classifier, op, gamma)

def build_bangrid(dataset, num_hid, op='', gamma=4, task='vqa'):
    conv_net = resnet_152_features()
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    v_att = BiAttention(2048, num_hid, num_hid, gamma)
    if task == 'vqa':
        b_net = []
        q_prj = []
        for i in range(gamma):
            b_net.append(BCNet(2048, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
        return BanModelGrid(conv_net, dataset, w_emb, q_emb, v_att, b_net, q_prj, classifier, op, gamma)
    elif task == 'flickr':
        return BanModel_flickr(w_emb, q_emb, v_att, op, gamma)

