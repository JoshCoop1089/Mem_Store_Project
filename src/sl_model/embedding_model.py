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
        self.emb_loss = exp_settings['emb_loss']
        self.contrastive_switch = exp_settings['switch_to_contrastive']
        self.use_lstm3 = False

        self.big_ol_zero = torch.tensor([0.0], device = self.device, requires_grad=False)

        # Basic Layers
        if self.mem_mode == "one_layer" or self.mem_mode == "two_layer":
            self.h2m = nn.Linear(
                self.input_dim, 2 * self.embedding_size, bias=bias, device=device
            )		
            if self.mem_mode == "two_layer":
                self.m2c = nn.Linear(
                    2*self.embedding_size, self.embedding_size, bias=bias, device = device
                )

        self.e2c = nn.Linear(
            self.embedding_size//2, self.num_barcodes, bias=bias, device=device
        )

        # LSTM2 Core
        if self.mem_mode == "LSTM":
            self.LSTM = nn.LSTM(input_size = self.input_dim, hidden_size = self.embedding_size, device = self.device)
            self.h_lstm, self.c_lstm = self.emb_get_init_states(self.embedding_size)
            self.l2i = nn.Linear(
                self.embedding_size, self.embedding_size//2, bias=bias, device=device
            )

            # LSTM3 Core (Contrastive Loss Post K-Means Loss)
            self.LSTM3 = nn.LSTM(input_size = self.embedding_size//2, hidden_size = self.embedding_size//4, device = self.device)
            self.h_lstm3, self.c_lstm3 = self.emb_get_init_states(self.embedding_size//4)
            self.l32i = nn.Linear(
                self.embedding_size//4, self.embedding_size//4, bias=bias, device=device
            )
            if self.emb_loss == 'contrastive' and self.contrastive_switch:
                self.use_lstm3 = True

        # init
        self.reset_parameter()

    def emb_get_init_states(self, lstm_hidden_dim, scale=0.1):
        h_0 = torch.randn(1, lstm_hidden_dim, device=self.device) * scale
        c_0 = torch.randn(1, lstm_hidden_dim, device=self.device) * scale
        return h_0, c_0

    # Model should return an embedding and a context
    def forward(self, h):
        if self.mem_mode == 'one_layer':
            x = self.h2m(h)
        elif self.mem_mode == 'two_layer':
            x = self.h2m(h)
            x = nn.Dropout(self.dropout_coef)(x)
            x = self.m2c(F.leaky_relu(x))
        elif self.mem_mode == 'LSTM':
            # K-Means -> Cross Entropy Trained LSTM
            x, (h1,c1)  = self.LSTM(h, (self.h_lstm,self.c_lstm))
            self.h_lstm, self.c_lstm = h1,c1
            x = self.l2i(F.leaky_relu(x))

            # Contrastive Loss Trained LSTM
            if self.use_lstm3:
                x = F.leaky_relu(x)
                x, (h3,c3)  = self.LSTM(x, (self.h_lstm3,self.c_lstm3))
                self.h_lstm3, self.c_lstm3 = h3,c3
                x = self.l32i(F.leaky_relu(x))

        else:
            raise ValueError("Incorrect mem_mode spelling")
        
        embedding = x

        if self.exp_settings['emb_loss'] != 'contrastive':
            predicted_context = self.e2c(F.leaky_relu(x))
        else:
            predicted_context = self.big_ol_zero

        return embedding, predicted_context

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            # print("emb: ", name)
            if "weight" in name:
                torch.nn.init.orthogonal_(wts)
            elif "bias" in name:
                torch.nn.init.constant_(wts, 0)
