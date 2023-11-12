import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt


SOS = 1
EOS = 0
class EncoderRNN(nn.Module):

    def __init__(self, input_size, device, hidden_size, dropout_p=0.1, emb = True):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size).to(self.device)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(dropout_p).to(self.device)



    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # analizza ogni riga della matrice, per tutta la sua lunghezza
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        if target_tensor is not None:
          dim = target_tensor.shape[1]
        else:
          dim = max_length + 1

        for i in range(dim):
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

def numpy2torch(numpy_path):
    """
    It converts the numpy matrix into a torch tensor to process the data
    """
    matrix = np.load(numpy_path)

    return matrix

def torch_format(batch_size, input_train, output_train, device):

  train_data = TensorDataset(torch.LongTensor(input_train).to(device), torch.LongTensor(output_train).to(device))
  #train_sampler = RandomSampler(train_data)
  #train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  train_dataloader = DataLoader(train_data, batch_size=batch_size)

  return train_dataloader


def train_epoch(encoder, decoder, n_elem_batch, learning_rate, train_dataloader, loss_function, device):
    """
    training funtion on a single epoch of the input matrix dataset
    """
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    total_loss_batch = 0

    for data in train_dataloader:

      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()

      input_tensor, output_tensor = data

      encoder_outputs, encoder_hidden = encoder(input_tensor)
      decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, output_tensor)

      loss = loss_function(
            decoder_outputs.view(-1, decoder_outputs.size(-1)).to(device),
            output_tensor.view(-1).to(device) )

      loss.backward()
      total_loss_batch += loss.item()

      encoder_optimizer.step()
      decoder_optimizer.step()

    return total_loss_batch/input_tensor.shape[0]


def train(encoder, decoder, n_elem_batch, learning_rate, train_dataloader, n_epochs, device):

    loss_function = nn.NLLLoss()
    total_train_loss_plot = []


    for steps in tqdm( list (range(n_epochs)), desc="number of epochs"):

        error = train_epoch(encoder, decoder, n_elem_batch, learning_rate, train_dataloader, loss_function, device)
        total_train_loss_plot.append(error)

    plt.plot(total_train_loss_plot)

def test(encoder, decoder, test_dataloader, device):
    encoder.eval()
    decoder.eval()

    loss_function = nn.NLLLoss()
    test_loss = 0

    with torch.no_grad():
      for data in test_dataloader:
          input_tensor, output_tensor = data
          encoder_outputs, encoder_hidden = encoder(input_tensor)
          decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, output_tensor)

          test_loss += loss_function(
          decoder_outputs.view(-1, decoder_outputs.size(-1)).to(device),
          output_tensor.view(-1).to(device))

          
      return test_loss / decoder_outputs.shape[0]

def translation(encoder, decoder, input_lang, output_lang,device):

    input_sentence = input("Type the sentence you want to translate:")
    vector = input_lang.string_translation(input_sentence)
    input_tensor = torch.tensor(vector,dtype=torch.long).to(device).unsqueeze(1)

    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        for i in range(3):
          decoded_words = []
          print(f"{i} solution proposed out of 3")
          for idx in decoded_ids[i]:
            if idx.item() == EOS:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])

          print(f"Original sentence = {input_sentence} --> Translated sentence = {decoded_words}")


def evaluation(encoder, decoder, n_elem_batch, learning_rate, n_epochs, train_dataloader, device, test_dataloader):

    train(encoder, decoder, n_elem_batch, learning_rate, train_dataloader, n_epochs,device)
    test(encoder, decoder, test_dataloader, device)

