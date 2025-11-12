import torch.nn as nn
from transformers import AutoModel
import torch

# Kích thước embedding của PhoBERT
PHOBERT_EMBEDDING_DIM = 768

# Định nghĩa lại class PhoBERT_CNN_GRU_Sentiment (3 nhãn: Tiêu cực, Trung tính, Tích cực)
class PhoBERT_CNN_GRU_Sentiment(nn.Module):
    def __init__(self, phobert, n_classes=3):
        super().__init__()
        self.phobert = phobert
        self.conv1 = nn.Conv1d(PHOBERT_EMBEDDING_DIM, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # GRU bidirectional, output 2*128=256
        self.gru = nn.GRU(256, 128, batch_first=True, bidirectional=True) 
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes) 

    def forward(self, input_ids, attention_mask):
        # Không dùng torch.no_grad() ở đây vì ta sẽ gọi nó khi dự đoán
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        x = self.relu(self.conv1(x.permute(0,2,1)))
        x = x.permute(0,2,1)
        _, h = self.gru(x) # h shape: (2, batch_size, 128)
        # Concatenate hidden states từ hai chiều
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1) 
        return self.fc(self.dropout(h))

# Định nghĩa lại class PhoBERT_GRU_Topic (4 nhãn: Giảng viên, Chương trình,...)
class PhoBERT_GRU_Topic(nn.Module):
    def __init__(self, phobert, n_classes=4):
        super().__init__()
        self.phobert = phobert
        self.gru = nn.GRU(PHOBERT_EMBEDDING_DIM, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        _, h = self.gru(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return self.fc(self.dropout(h))