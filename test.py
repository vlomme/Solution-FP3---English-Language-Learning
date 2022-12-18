# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import DataCollatorWithPadding
import os
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Data Loading"""
if os.path.exists("settings.json"):
    with open("settings.json") as f:
        settings = json.load(f)
    train = settings["TRAIN_DATA_CLEAN_PATH"]
    psevdo_old = settings["TRAIN_DATA_CLEAN_PATH_OLD"]
    psevdo_new = settings["TRAIN_DATA_CLEAN_PATH_NEW"] 
    chk = settings["MODEL_CHECKPOINT_DIR"]

"""# Dataset"""

# ====================================================
# Dataset
# ====================================================
def prepare_input(tokenizer, text):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=512,
        #pad_to_max_length=True,
        truncation=True,return_token_type_ids=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.tokenizer, self.texts[item])
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
    def __init__(self, model_name=None, pretrained=False,out=7,att = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.att = att
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, out)
        self._init_weights(self.fc)
        self.Sigmoid = nn.Sigmoid()
        if self.att:
            self.attention = AttentionPool(self.config.hidden_size)
        
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
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        if self.att:
            feature = self.attention(last_hidden_states, inputs['attention_mask'])     
        else:
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs, bce):
        feature = self.feature(inputs)
        output = self.fc(feature)[:,:6]
        if bce:
            output = self.Sigmoid(output)*4 + 1
            output = output - 3
        return output,feature

"""# inference"""

# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device,bce):
    preds = []
    model.eval()
    model.to(device)
    for inputs in test_loader:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds,feature = model(inputs,bce)
        preds.append(y_preds.to('cpu').numpy())
    tmp_predictions = np.concatenate(preds)
    return tmp_predictions

args = {}
args[0] = {
    "model" : 'microsoft/deberta-v3-xsmall',
    "batch_size" : 1,
    "out":6,
    "bce":False,
    "att":False,
    'fold' : [
                chk,
              ],
    "k" : 1
}

if __name__ == '__main__': 
    test = pd.read_csv(settings["TEST_DATA_CLEAN_PATH"])
    for model_id in [0]:
        tokenizer = AutoTokenizer.from_pretrained(args[model_id]['model'])
        test_dataset = TestDataset(tokenizer, test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args[model_id]['batch_size'],
                                 shuffle=False,
                                 collate_fn=DataCollatorWithPadding(tokenizer, padding='longest'),
                                 num_workers=4, pin_memory=True, drop_last=False)

        for fold in args[model_id]['fold']:
            model = CustomModel(model_name = args[model_id]['model'], pretrained=False, out = args[model_id]['out'], att=args[model_id]['att'])#
            state = torch.load(fold, map_location=torch.device('cpu'))
            model.load_state_dict(state['model'])
            model.float()
            prediction = inference_fn(test_loader, model, device,args[model_id]['bce'])  + 3

    """# Submission"""

    EPS = 0.015
    snap = (np.array(prediction) * 2).round() / 2
    err = snap - np.array(prediction)
    to_snap = np.abs(err) < EPS
    prediction = np.where(to_snap, snap, np.array(prediction))

    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    test[target_cols] = prediction
    if not os.path.exists(settings["SUBMISSION_DIR"]):
        os.makedirs(settings["SUBMISSION_DIR"])
    test[['text_id'] + target_cols].to_csv(settings["SUBMISSION_DIR"]+'submission.csv', index=False)
