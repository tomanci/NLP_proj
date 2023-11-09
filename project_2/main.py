from text_process import *
from RNN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_dataset = "/Users/tommasoancilli/Downloads/ita-eng/ita2eng.txt"
max_length = 10 

Text_creation (path_dataset, epsilon = 0.3, max_length = max_length)
input_lang = Language(path_name = "eng.txt")
output_lang = Language(path_name = "ita.txt")   

input_lang.string_processing()
output_lang.string_processing()

Dataset_creation(input_lang, output_lang,training_val_test_ratio= (0.85,0.05,0.1))

hidden_size = 128 #64 
batch_size = 32

input_train = numpy2torch("input_train.npy")
output_train = numpy2torch("output_train.npy")

input_val = numpy2torch("input_val.npy")
output_val = numpy2torch("output_val.npy")

input_test = numpy2torch("input_test.npy")
output_test = numpy2torch("output_test.npy")

Encoder = EncoderRNN(input_size=input_lang.n_words, hidden_size=hidden_size).to(device)
Decoder = DecoderRNN(hidden_size=hidden_size, output_size=output_lang.n_words).to(device)

evaluation(encoder=Encoder, decoder=Decoder, n_elem_batch=batch_size, learning_rate=0.001,n_epochs=32, 
           input_train=input_train, output_train=output_train, input_val=input_val, output_val=output_val, input_test=input_test, output_test=output_test)