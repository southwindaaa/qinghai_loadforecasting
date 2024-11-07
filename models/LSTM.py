from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from peft import LoraConfig, get_peft_model

transformers.logging.set_verbosity_error()




class Model(nn.Module):
    def __init__(self, configs, social_vocab_size=100, social_embedding_dim=16):
        super(Model, self).__init__()
        # LSTM部分
        hidden_size = 128
        self.lstm = nn.LSTM(configs.enc_in+social_embedding_dim, hidden_size, 2, batch_first=True)
        
        # Social 信息的 embedding 层
        self.social_embedding = nn.Embedding(social_vocab_size, social_embedding_dim)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, configs.pred_len)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, social_ids):
        # x_enc: 时间序列数据 [batch_size, seq_len, input_size]
        # social_ids: 社会信息 (类别ID) [batch_size]，这里每个样本对应一个社会信息类别ID
        
        # 获取 social 信息的 embedding
        social_embeddings = self.social_embedding(social_ids)  # [batch_size, social_embedding_dim]
        
        # 将 social 信息嵌入扩展到每个时间步上
        social_embeddings = social_embeddings.unsqueeze(1).expand(-1, x_enc.size(1), -1)  # [batch_size, seq_len, social_embedding_dim]
        # 将时间序列输入和 social 嵌入拼接
        lstm_input = torch.cat((x_enc, social_embeddings), dim=2)  # [batch_size, seq_len, input_size + social_embedding_dim]
        
        # 输入 LSTM
        lstm_out, _ = self.lstm(lstm_input)  # [batch_size, seq_len, hidden_size]
        lstm_out = lstm_out[:, -1:, :]  # 取最后一个时间步的输出 [batch_size, hidden_size]
        # 预测输出
        output = self.fc(self.dropout(lstm_out))  # [batch_size, output_size]
        return output

