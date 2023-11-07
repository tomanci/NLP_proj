import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

SOS = 1
class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, dropout_p=0.1, emb = True):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
            

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, MAX_LENGTH, target_tensor=None):
        # analizza ogni riga della matrice, per tutta la sua lunghezza 
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

def _numpy2torch(numpy_path):
    """
    It converts the numpy matrix into a torch tensor to process the data 
    """    
    matrix = np.load_matrix(numpy_path)

    return torch.tensor(matrix)

def train_epoch(encoder, decoder, n_elem_batch, learning_rate, input_train:torch.tensor, output_train:torch.tensor, loss_function):
    """
    training funtion on a single epoch of the input matrix dataset
    """
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_oprimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    n_batch = input_train.shape[0] // n_elem_batch #quotient
    elem_last_batch = input_train.shape[0] % n_elem_batch #resto
    total_loss_batch = 0 

    for i in range(n_batch):
        
        encoder_optimizer.zero_grad()
        decoder_oprimizer.zero_grad()
        
        input_tensor = input_train[i*n_elem_batch: (i+1)*n_elem_batch, : ].shape(1,-1)
        #in base alla funzione, al fatto che l'encoder prende in input un vettore e non una matrice ( tensor Dataset)
        output_tensor = output_train[i*n_elem_batch: (i+1)*n_elem_batch, : ].shape(1,-1)
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, output_tensor)

        loss = loss_function(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            output_tensor.view(-1))
        
        loss.backward()
        total_loss_batch += loss.item()

        encoder_optimizer.step()
        decoder_oprimizer.step()

        #last elements of the training set
        encoder_optimizer.zero_grad()
        decoder_oprimizer.zero_grad()

        input_tensor = input_train[(i+1)*n_elem_batch:, : ].shape(1,-1)
        #in base alla funzione, al fatto che l'encoder prende in input un vettore e non una matrice ( tensor Dataset)
        output_tensor = output_train[(i+1)*n_elem_batch:, : ].shape(1,-1)


        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, output_tensor)

        loss = loss_function(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            output_tensor.view(-1))
        
        loss.backward()
        total_loss_batch += loss.item()

        encoder_optimizer.step()
        decoder_oprimizer.step()



# DA FARE VALIDATION, TESTING FUNCTION E TRAINING FUNCTION CHE CONSIDERA LE EPOCHE 




