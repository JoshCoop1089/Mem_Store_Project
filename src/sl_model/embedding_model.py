from numpy import int8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, exp_settings, device, bias=True):
        super(Embedder, self).__init__()
        self.bias = bias
        self.exp_settings = exp_settings
        self.device = device
        self.input_dim = exp_settings["dim_hidden_lstm"]
        self.embedding_size = exp_settings["embedding_size"]
        self.num_barcodes = exp_settings["num_barcodes"]
        self.dropout_coef = exp_settings["dropout_coef"]

        self.mem_mode = exp_settings["mem_mode"]

        # Basic Layers
        self.h2m = nn.Linear(
            self.input_dim, 2 * self.embedding_size, bias=bias, device=device
        )		
        self.m2c = nn.Linear(
            2*self.embedding_size, self.embedding_size, bias=bias, device = device
        )
        self.e2c = nn.Linear(
            self.embedding_size, self.num_barcodes, bias=bias, device=device
        )

        self.LSTM = nn.LSTM(input_size = self.input_dim, hidden_size = self.embedding_size, device = self.device)
        self.h_lstm, self.c_lstm = self.emb_get_init_states()

        # init
        self.reset_parameter()

    def emb_get_init_states(self, scale=0.1):
        h_0 = torch.randn(1, self.embedding_size, device=self.device) * scale
        c_0 = torch.randn(1, self.embedding_size, device=self.device) * scale
        return h_0, c_0

    # Model should return an embedding and a context
    def forward(self, h):
        if self.mem_mode == 'two_layer':
            x = self.h2m(h)
            x = nn.Dropout(self.dropout_coef)(x)
            x = self.m2c(F.leaky_relu(x))
        elif self.mem_mode == 'LSTM':
            x, (h1,c1)  = self.LSTM(h, (self.h_lstm,self.c_lstm))
        self.h_lstm, self.c_lstm = h1,c1
        embedding = x
        predicted_context = self.e2c(F.leaky_relu(x))

        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print("emb: ", name)
            if "weight" in name:
                torch.nn.init.orthogonal_(wts)
            elif "bias" in name:
                torch.nn.init.constant_(wts, 0)
