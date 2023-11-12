from text_process import *
from RNN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path_dataset = "/Users/tommasoancilli/Downloads/ita-eng/ita2eng.txt"
max_length = 10

Text_creation(path_dataset, epsilon = 0.3, max_length = max_length)
input_lang = Language(path_name = "eng.txt")
output_lang = Language(path_name = "ita.txt")

input_lang.string_processing()
output_lang.string_processing()

Dataset_creation(input_lang, output_lang,training_test_ratio= (0.85,0.15))

hidden_size = 1024
batch_size = 64
number_of_epochs = 256

input_train = numpy2torch("input_train.npy")
output_train = numpy2torch("output_train.npy")

input_test = numpy2torch("input_test.npy")
output_test = numpy2torch("output_test.npy")

Encoder = EncoderRNN(input_size=input_lang.n_words, device=device, hidden_size=hidden_size).to(device)
Decoder = DecoderRNN(hidden_size=hidden_size, device = device, output_size=output_lang.n_words).to(device)

train_dataloader = torch_format(batch_size, input_train, output_train, device)
test_dataloader = torch_format(batch_size, input_test, output_test, device)

evaluation(encoder=Encoder, decoder=Decoder, n_elem_batch=batch_size, learning_rate=0.001, n_epochs=number_of_epochs,
           train_dataloader = train_dataloader, test_dataloader = test_dataloader, device= device)

while True:
  translation(encoder=Encoder, decoder=Decoder, input_lang=input_lang, output_lang=output_lang, device=device)
  response = input("do you want to translate other sentences? [Y/N]: ")
  if response == "N":
    break
