import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import torch
import random
from torch.utils.data.dataloader import default_collate

class imsitu_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, roles, labels = self.encoder.encode(ann)

        return _id, img, verb, roles, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb = self.encoder.encode(ann)

        return _id, img, verb

    def __len__(self):
        return len(self.annotations)


class imsitu_loader_agentverbq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb = self.encoder.encode(ann)

        return _id, img, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_agent2verbqpure(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verbq = self.encoder.encode(ann, _id)

        return _id, img, verb, verbq

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_agentverbq_roles(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, labels = self.encoder.encode(ann)

        return _id, img, verb, labels

    def __len__(self):
        return len(self.annotations)


class imsitu_loader_roleq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, roles, roleq, q_len, labels = self.encoder.encode(ann)
        return _id, img, verb, roles, roleq, q_len, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_roleq_updated(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, labels = self.encoder.encode_verb(ann)
        return _id, img, verb, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_hico(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb = torch.tensor(ann).type(torch.FloatTensor)
        return _id, img, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verbq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verb_q, roles, labels = self.encoder.encode(ann, _id)
        return _id, img, verb, verb_q, roles, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verbq_mul(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb = self.encoder.encode(ann, _id)
        return _id, img, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_place4verbq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verbq = self.encoder.encode(ann, _id)
        return _id, img, verb, verbq

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_rotation(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.first_transform = transform
        self.second_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        _id = self.ids[index]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        trans_1 = self.first_transform(img)
        rotated_imgs = [
            self.second_transform(trans_1),
            self.second_transform(rotate_img(trans_1,  90)),
            self.second_transform(rotate_img(trans_1, 180)),
            self.second_transform(rotate_img(trans_1, 270))
        ]
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return torch.stack(rotated_imgs, dim=0), rotation_labels


    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb_roleq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verbq, labels = self.encoder.encode(ann, _id)
        return _id, img, verb, verbq, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb_roleq_obj(data.Dataset):
    def __init__(self, img_dir, annotation_file, obj_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform
        self.obj_det = obj_file

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        detected_obj = self.obj_det[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verbq, labels = self.encoder.encode(ann)
        return _id, img, torch.tensor(detected_obj, dtype=torch.float), verb, verbq, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb_roleq_phrase(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, verbq, roleq, roles, labels = self.encoder.encode(ann)
        return _id, img, verb, verbq, roleq, roles, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_frcnn(data.Dataset):
    def __init__(self, img_dir, frcnn_feat_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.frcnn_feat_dir = frcnn_feat_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        #print('current idx :' , index)
        _id = self.ids[index]
        '''if index % 2 == 0:
            _id = 'pitching_187.jpg'
        else:
            _id = 'shouting_260.jpg'''
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        frcnn_feat = torch.from_numpy(np.load(os.path.join(self.frcnn_feat_dir,
                                                           _id.replace('jpg', 'npy'))))
        verb, roles, labels = self.encoder.encode(ann)

        #print('feat loader:', frcnn_feat.size())

        return _id, img, frcnn_feat, verb, roles, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_frcnn_roleq(data.Dataset):
    def __init__(self, img_dir, frcnn_feat_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.frcnn_feat_dir = frcnn_feat_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        #print('current idx :' , index)
        _id = self.ids[index]
        '''if index % 2 == 0:
            _id = 'pitching_187.jpg'
        else:
            _id = 'shouting_260.jpg'''
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        frcnn_feat = torch.from_numpy(np.load(os.path.join(self.frcnn_feat_dir,
                                                           _id.replace('jpg', 'npy'))))
        verb, roles, roleq, q_len, labels = self.encoder.encode(ann)

        #print('feat loader:', frcnn_feat.size())

        return _id, img, frcnn_feat, verb, roles, roleq, q_len, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_agent(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        labels = self.encoder.encode(ann)

        if labels.size(0) != 3:
            print('ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR')
        return _id, img, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_agent_objoh(data.Dataset):
    def __init__(self, img_dir, annotation_file, obj_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform
        self.obj_det = obj_file

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        detected_obj = self.obj_det[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        labels = self.encoder.encode(ann)

        if labels.size(0) != 3:
            print('ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR')
        return _id, img, torch.tensor(detected_obj, dtype=torch.float), labels

    def __len__(self):
        return len(self.annotations)


def shuffle_minibatch(batch):
    random.shuffle(batch)
    ids, imgs, verbs, role_set, label_set = [],[],[],[],[]

    for i, b in enumerate(batch):
        _id, img, verb, roles, labels = b
        ids.append(_id)
        imgs.append(img)
        verbs.append(verb)
        role_set.append(roles)
        label_set.append(labels)

    return ids, torch.stack(imgs), torch.LongTensor(verbs), torch.stack(role_set), torch.stack(label_set)

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.ascontiguousarray(np.flipud(np.transpose(img, (1,0,2))))
    elif rot == 180: # 90 degrees rotation
        return np.ascontiguousarray((np.fliplr(np.flipud(img))))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.ascontiguousarray(np.transpose(np.flipud(img), (1,0,2)))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def _collate_fun(batch):
    batch = default_collate(batch)
    assert(len(batch)==2)
    batch_size, rotations, channels, height, width = batch[0].size()
    batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
    batch[1] = batch[1].view([batch_size*rotations])
    return batch