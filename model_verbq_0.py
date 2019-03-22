import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import utils_imsitu as utils
import numpy as np

from attention import BiAttention
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def rep_size(self):
        return 1024

    def base_size(self):
        return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        features = self.vgg_features(x)

        return features


class BanModel(nn.Module):
    def __init__(self, n_verbs, glimpse=4,
                 embed_hidden=300,
                 mlp_hidden=512):
        super(BanModel, self).__init__()
        self.n_verbs = n_verbs
        self.glimpse = glimpse
        self.q_emb = nn.LSTM(embed_hidden, mlp_hidden,
                             batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(mlp_hidden * 2, mlp_hidden)
        self.v_att = BiAttention(mlp_hidden, mlp_hidden, mlp_hidden, glimpse)
        b_net = []
        q_prj = []
        for i in range(glimpse):
            b_net.append(BCNet(mlp_hidden, mlp_hidden, mlp_hidden, None, k=1))
            q_prj.append(FCNet([mlp_hidden, mlp_hidden], '', .2))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)

        self.classifier = SimpleClassifier(
            mlp_hidden, mlp_hidden * 2, self.n_verbs, .5)

        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v,q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = q
        self.q_emb.flatten_parameters()
        q_emb, (h, _) = self.q_emb(w_emb)
        q_emb = self.lstm_proj(q_emb)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h

            atten, _ = logits[:,g,:,:].max(2)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        logits = self.classifier(q_emb.sum(1))

        return logits

class BaseModel(nn.Module):
    def __init__(self, encoder,
                 gpu_mode,
                 embed_hidden=300,
                 mlp_hidden = 512
                 ):
        super(BaseModel, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(10),
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoder = encoder
        self.gpu_mode = gpu_mode
        self.mlp_hidden = mlp_hidden
        self.verbq_word_count = len(self.encoder.verb_question_words)
        self.n_verbs = self.encoder.get_num_verbs()


        self.conv = vgg16_modified()


        self.verb_vqa = BanModel(self.n_verbs)
        self.verb_q_emb = nn.Embedding(self.verbq_word_count + 1, embed_hidden, padding_idx=self.verbq_word_count)



    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self, ):
        return self.dev_transform

    def forward(self, img, verbs=None, labels=None):

        verb_q_idx = self.encoder.get_common_verbq(img.size(0))

        if self.gpu_mode >= 0:
            verb_q_idx = verb_q_idx.to(torch.device('cuda'))

        img_embd = self.conv(img)
        batch_size, n_channel, conv_h, conv_w = img_embd.size()
        img_embd = img_embd.view(batch_size, n_channel, -1)
        img_embd = img_embd.permute(0, 2, 1)

        q_emb = self.verb_q_emb(verb_q_idx)

        verb_pred_logit = self.verb_vqa(img_embd, q_emb)

        return verb_pred_logit

    def calculate_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss