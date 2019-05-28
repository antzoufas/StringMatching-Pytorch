import torch
from torch import nn
from torchvision import transforms, datasets
from torch.autograd import Variable

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class Encoder(nn.Module):
    def __init__(self, hidden_units=60, bidirectional=True, \
                     highway=True, self_attention=True, max_pooling=True, alignment=True,\
                     shortcut=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_units
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU()
        self.gru1 = nn.GRU(195, hidden_size=self.hidden_size, batch_first=True, bidirectional=self.bidirectional)
        self.gru2 = nn.GRU(2*hidden_units, hidden_size=self.hidden_size, batch_first=True, bidirectional=self.bidirectional)
        self.trainable_layers = [self.gru1, self.gru2]

    def reset_parameters(self):
        for layer in self.trainable_layers:
            layer.reset_parameters()
    
    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1
        return Variable(torch.zeros((directions, batch_size, self.hidden_size))).cuda()
    
    def forward(self, s, lengths):
        # Zero initial hidden state for both GRUs
        hidden_1 = self.init_hidden(s.size()[0])
        hidden_2 = self.init_hidden(s.size()[0])

        # Dynamic sequence length
        s = torch.nn.utils.rnn.pack_padded_sequence(s, lengths, batch_first=True, enforce_sorted=False)

        s_rep, hidden_1 = self.gru1(s, hidden_1)
        s_rep, _ = torch.nn.utils.rnn.pad_packed_sequence(s_rep, batch_first=True)
        s_rep = self.relu(s_rep)
        s_rep = self.dropout(s_rep)

        s_rep = torch.nn.utils.rnn.pack_padded_sequence(s_rep, lengths, batch_first=True, enforce_sorted=False)
        _, s_rep = self.gru2(s_rep, hidden_2) # s_rep contains last hidden_state
        s_rep = self.relu(s_rep)
        s_rep = self.dropout(s_rep)

        # Convert (len, batch_size, features) into (batch_size, len, features)
        s_rep = s_rep.permute(1,0,2)
        s_rep = s_rep.contiguous()
        s_rep = s_rep.view(s_rep.size()[0], -1) # Concatenate forward and backward RNN
        return s_rep

class StringMatcher(nn.Module):
    def __init__(self, hidden_units=60, bidirectional=True, \
                     highway=True, self_attention=True, max_pooling=True, alignment=True,\
                     shortcut=True):
        super(StringMatcher, self).__init__()
        self.hidden_units = hidden_units
        self.lin1 = nn.Linear(self.hidden_units*8 if bidirectional else self.hidden_units*4, self.hidden_units)
        self.lin2 = nn.Linear(self.hidden_units, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.01)
        self.soft = nn.Softmax(dim=1)
        self.Encoder = Encoder(self.hidden_units, bidirectional, highway, self_attention, max_pooling, alignment, shortcut)
        self.trainable_layers = [self.lin1, self.lin2, self.Encoder]

    def reset_parameters(self):
        for layer in self.trainable_layers:
            layer.reset_parameters()
    
    def forward(self, s1, s2, s1_lens, s2_lens):
        # Representation for each input sentence
        s1_rep = self.Encoder(s1, s1_lens)
        s2_rep = self.Encoder(s2, s2_lens)

        # Concatenate, multiply, subtract
        conc = torch.cat([s1_rep, s2_rep], 1)
        mul = s1_rep * s2_rep
        dif = torch.abs(s1_rep - s2_rep)
        final = torch.cat([conc, mul, dif], 1)
        final = self.dropout(final)

        # Linear layers and softmax
        final = self.lin1(final)
        final = self.relu(final)
        final = self.dropout(final)
        final = self.lin2(final)
        out = self.soft(final)
        return out

def load_model(hidden_units, bidirectional, highway, self_attention, max_pooling, alignment, shortcut):
    return StringMatcher(hidden_units, bidirectional, highway, self_attention, max_pooling, alignment, shortcut).cuda()
