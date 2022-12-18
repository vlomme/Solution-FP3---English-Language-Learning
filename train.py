# -*- coding: utf-8 -*-

# ====================================================
# Directory settings
# ====================================================
import os
import os.path
import json
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

"""# CFG"""

# ====================================================
# CFG
# ====================================================
class CFG:
    competition='FB3'
    debug=False
    apex=True
    num_workers=0 
    model = "microsoft/deberta-v3-xsmall"
    gradient_checkpointing=True
    scheduler='polynomial' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=.5
    num_warmup_steps=0.1
    epochs=2
    encoder_lr=6e-5
    decoder_lr=1e-4
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=32  
    print_freq = [(x * 4800)//32 for x in range(1,100)]
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=15
    trn_fold= [5]
    train=True
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

"""# Library"""

# Commented out IPython magic to ensure Python compatibility.
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")
import argparse
import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset


from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
# %env TOKENIZERS_PARALLELISM=true

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
from sklearn import model_selection
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def create_folds(data, num_splits,seed):
    data["kfold"] = -1

    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data

"""# Utils"""

# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]# - 1
    if idxes==7:
        idxes = idxes - 1
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--data", type=str, required=False, default="./data")
    parser.add_argument("--chk", type=str, required=False, default="")
    parser.add_argument("--mod", type=str, required=False, default="train")
    parser.add_argument("--typ", type=str, required=False, default="all")
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--output", type=str, default=".", required=False)
    parser.add_argument("--input", type=str, default="./", required=False)
    parser.add_argument("--max_len", type=int, default=512, required=False)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=4, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--predict", action="store_true", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    return parser.parse_args()
    
def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    scores = [round(elem, 3) for elem in scores]
    return mcrmse_score, scores
def monitor_metrics(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss

def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    



# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, tokenizer, text):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, tokenizer, df):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.tokenizer, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

"""# Model"""

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, x, mask):
        w = self.attention(x).float() #
        w[mask==0]=float('-inf')
        w = torch.softmax(w,1)
        x = torch.sum(w * x, dim=1)
        return x
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            self.config.use_cache=False
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        #print(self.model)
        if 'deberta-v2-xxlarge' in cfg.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)
        if 'deberta-v3-large' in cfg.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:11].requires_grad_(False)
            print(11,len(self.model.encoder.layer))
        if  'deberta-large' in cfg.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)
            print(12,len(self.model.encoder.layer))
        #if 'roberta-large' in cfg.model:
        #    #self.model.embeddings.requires_grad_(False)
        #    self.model.encoder.layer[:4].requires_grad_(False)
        #    print(4,len(self.model.encoder.layer))
        if 'deberta-v2-xlarge' in cfg.model or 'deberta-xlarge' in cfg.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:23].requires_grad_(False)
        if 'facebook/bart-large' in cfg.model:
            #self.model.embeddings.requires_grad_(False)
            self.model.encoder.layers[:4].requires_grad_(False)
            print(4,len(self.model.encoder.layers))
        if 'xlm-roberta-large' in cfg.model:
            #self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:20].requires_grad_(False) 
            print(20,len(self.model.encoder.layer))
        if 'funnel-transformer-xlarge' in cfg.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False)
        
        #self.attention = AttentionPool(self.config.hidden_size)
        self.criterion1 = torch.nn.BCEWithLogitsLoss()
        self.criterion = nn.SmoothL1Loss(reduction='mean')
        self.fc = nn.Linear(self.config.hidden_size, len(cfg.target_cols))
        self.Sigmoid = nn.Sigmoid()
        #self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)#[1]
        last_hidden_states = outputs[-1][-1]
        #last_hidden_states = torch.cat(outputs[-3:], 2)
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        #feature = self.attention(last_hidden_states, inputs['attention_mask'])
        #feature = last_hidden_states[:,0]
        #y = torch.mean(last_hidden_states, 1) 
        return feature

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        output = self.fc(feature)
        loss = self.criterion(output, labels-3)
        output = output + 3
        #loss = monitor_metrics(output, labels)
        #loss = self.criterion1(output, (labels-1)/4)
        #output = self.Sigmoid(output)*4 + 1
        return output, feature, loss


"""# Loss"""


"""# Helpler functions"""

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, valid_loader, valid_labels, best_score, model,  optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        
        for k, v in inputs.items():
            inputs[k] = v.to(device)
            
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds, _, loss = model(inputs, labels)
            
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        #grad_norm = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step >0 and (step in CFG.print_freq  or step == (len(train_loader)-1)):
            #if (step >= 500 and step%300 == 0) or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

            if epoch>0  or CFG.epochs == 1:
                # eval
                avg_val_loss, predictions = valid_fn(valid_loader, model, device)
                
                # scoring
                score, scores = get_score(valid_labels, predictions)

                LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f} avg_train_loss: {losses.avg:.4f} avg_val_loss: {avg_val_loss:.4f} Scores: {scores}')

                model.half()
                if best_score > score:
                    
                    best_score = score
                    LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                    torch.save({'model': model.state_dict(),
                                'predictions': predictions},
                                OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
                    
                if step == (len(train_loader)-1):
                    torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                            OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_{score}.pth")
            
                model.float()
                model.train()
    
    return best_score


def valid_fn(valid_loader, model,  device):
    losses = AverageMeter()
    model.eval()
    preds = []
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds,_, loss = model(inputs, labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return losses.avg, predictions
    
    
def create_optimizer(model, args):
    LR = args.lr
    named_parameters = list(model.named_parameters())    
    #print(model)
    print(LR, len(named_parameters))
    
    roberta_parameters = named_parameters[:-1]    
    #attention_parameters = named_parameters[-5:-1]
    regressor_parameters = named_parameters[-1:]
        
    #attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    #parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})
    k = len(roberta_parameters)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = LR/20
        
        if layer_num >= k/4:        
            lr = LR/10

        if layer_num >= k*2/4:
            lr = LR/5
        
        if layer_num >= k*3/4:
            lr = LR/2

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})
    print(k, layer_num)
    return AdamW(parameters,lr=4e-4)
"""# train loop"""

# ====================================================
# train loop
# ====================================================
def train_loop(tokenizer, folds, fold, args):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['kfold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['kfold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = TrainDataset(CFG, tokenizer, train_folds)
    valid_dataset = TrainDataset(CFG, tokenizer, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    if len(CFG.chk):
        state = torch.load(CFG.chk, map_location=torch.device('cpu'))
        model.load_state_dict(state['model'])
    
    reinit_layers = 0
    if reinit_layers > 0:
        print(f'Reinitializing Last {reinit_layers} Layers ...')
        #encoder_temp = getattr(model, 'deberta')
        for layer in model.model.encoder.layer[-reinit_layers:]:    
            for module in layer.modules():
                model._init_weights(module)
        print('Done.!')

    optimizer = create_optimizer(model, args) 
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(CFG, optimizer, num_train_steps):
        if CFG.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(CFG.num_warmup_steps * num_train_steps), num_training_steps=num_train_steps
            )
        elif CFG.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(CFG.num_warmup_steps * num_train_steps), num_training_steps=num_train_steps, num_cycles=CFG.num_cycles
            )
        elif CFG.scheduler == 'polynomial':
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps=int(CFG.num_warmup_steps * num_train_steps), num_training_steps=num_train_steps, 
                lr_end = 1e-6, power=2)    
        return scheduler

    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================

    best_score = np.inf
    start_time = time.time()
    for epoch in range(CFG.epochs):
        best_score = train_fn(fold, train_loader, valid_loader, valid_labels, best_score, model, optimizer, epoch, scheduler, device)
    print("прошло",time.time() - start_time)

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

if __name__ == '__main__':
    args = parse_args()
    CFG.path = str(args.seed) +"_"+ str(args.max_len) + "_"+ args.typ+"_"+CFG.model.replace('/', '-')
    OUTPUT_DIR = OUTPUT_DIR + CFG.path + "/"
    if not os.path.exists(CFG.path):
        os.makedirs(CFG.path)
    CFG.max_len = args.max_len
    CFG.epochs = args.epochs
    CFG.model = args.model
    CFG.chk = args.chk
    psevdo_old = args.data+'/psevdo_old.csv'
    train = args.data+'/train_folds.csv'
    psevdo_new = args.data+'/psevdo_new.csv'
    if os.path.exists("settings.json"):
        with open("settings.json") as f:
            settings = json.load(f)
        train = settings["TRAIN_DATA_CLEAN_PATH"]
        psevdo_old = settings["TRAIN_DATA_CLEAN_PATH_OLD"]
        psevdo_new = settings["TRAIN_DATA_CLEAN_PATH_NEW"] 

    
    print("seed",args.seed,args.max_len)
    seed_everything(seed=args.seed)
         
    if args.typ == "all":
        CFG.target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        #CFG.target_cols.append("m")
    else:
        CFG.target_cols = [args.typ]

    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

    """# tokenizer"""
    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
    
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train = pd.read_csv(train)
                train["m"] = np.mean(train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values, axis=1)
                
                psevdo = pd.read_csv(psevdo_old)
                psevdo['kfold'] = 99
                psevdo["m"] = np.mean(psevdo[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values, axis=1)

                
                train_fold0 = pd.read_csv(psevdo_new)
                train_fold0['kfold'] = 99
                train_fold0["m"] = np.mean(train_fold0[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values, axis=1)
                
                if args.mod == "psevdo_train":
                    train0 = train[train["kfold"] == fold]
                    ids = train0.text_id.unique()
                    #train_fold0 = train_fold0[~train_fold0.text_id.isin(ids)]
                    train = pd.concat([train0, train_fold0])
                if args.mod == "psevdo_old":
                    train0 = train[train["kfold"] == fold]
                    train = pd.concat([train0, psevdo])
                    #ids = train0.text_id.unique()
                    #train_fold0 = train_fold0[~train_fold0.text_id.isin(ids)]
                    #train = pd.concat([train, train_fold0,train_fold0, psevdo])
      
                print(f"train.shape: {train.shape}")
                print(train)
                _oof_df = train_loop(tokenizer, train, fold, args)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')