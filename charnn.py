import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # DONE:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    list_chars = list(set([char for char in text]))
    list_chars = sorted(list_chars, key=str.upper)
    char_to_idx = {c: idx for idx, c in enumerate(list_chars)}
    idx_to_char = {idx: c for c, idx in char_to_idx.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # DONE: Implement according to the docstring.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    text_clean = text
    for c in chars_to_remove:
        text_clean = text_clean.replace(c, '')

    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed

def idx_to_onehot(list_of_idx, D):
    N = len(list_of_idx)
    onehot_vec = torch.zeros((N, D), dtype=torch.int8)
    onehot_vec[torch.arange(N), list_of_idx] = 1
    return onehot_vec

def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # DONE: Implement the embedding.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    list_of_idx = [char_to_idx[c] for c in text]

    D = len(char_to_idx)
    result = idx_to_onehot(list_of_idx, D)
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # DONE: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    list_of_idx = (embedded_text).nonzero()[:, 1]
    list_of_chars = [idx_to_char[idx.item()] for idx in list_of_idx]
    result = ''.join(list_of_chars)
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    N = (len(text) - 1) // seq_len
    V = len(char_to_idx)
    embedded_text = chars_to_onehot(text, char_to_idx)[:(N * seq_len + 1)]
    samples = embedded_text[:-1].view((N, seq_len, V))

    labels = (embedded_text[1:]).nonzero()[:, 1]
    labels = labels.view((N, seq_len))
    # ========================
    samples = samples.to(device)
    labels = labels.to(device)
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    exp = torch.exp(y / temperature)
    result = exp / exp.sum(dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    h0 = None
    x = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).to(device)
    with torch.no_grad():
        for i in range(n_chars - len(start_sequence)):
            y, h0 = model(x.to(dtype=torch.float), h0)
            prob = hot_softmax(y[0][-1], dim=0, temperature=T)
            char_idx = torch.multinomial(prob, num_samples=1)
            char = idx_to_char[char_idx.item()]
            x = chars_to_onehot(char, char_to_idx).unsqueeze(0).to(device)
            out_text += char
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # DONE:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents  one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of indices is takes, samples in the same index of
        #  adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        # idx = None  # idx should be a 1-d list of indices.
        idx = []
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        batch_num = len(self.dataset) // self.batch_size
        N = batch_num * self.batch_size
        for i in range(batch_num):
            idx.extend(list(range(i, N, batch_num)))
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)

class GruBlock(nn.Module):
    def __init__(self, in_dim, h_dim,p):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.layer_params = []
        self.sig_z = torch.sigmoid
        self.sig_r = torch.sigmoid
        self.tanh_g = torch.tanh
        self.drop_out = torch.nn.Dropout(p)
        self.Wxz = nn.Linear(self.in_dim, self.h_dim,   bias=False)
        self.Wxr = nn.Linear(self.in_dim, self.h_dim,   bias=False)
        self.Wxg = nn.Linear(self.in_dim, self.h_dim,   bias=False)
        self.Whz_bz = nn.Linear(self.h_dim, self.h_dim,   bias=True)
        self.Whr_br = nn.Linear(self.h_dim, self.h_dim,   bias=True)
        self.Whg_bg = nn.Linear(self.h_dim, self.h_dim,   bias=True)

    def forward(self, input: Tensor, hidden_state: Tensor ):
        z_t = self.sig_z(self.Wxz(input)+self.Whz_bz(hidden_state))
        r_t = self.sig_r(self.Wxr(input)+self.Whr_br(hidden_state))
        g_t = self.tanh_g(self.Wxg(input)+self.Whg_bg(r_t*hidden_state))
        h_t = z_t*hidden_state+(1-z_t)*g_t
        h_t_drop_out = self.drop_out(h_t)
        return h_t, h_t_drop_out

class GruColumn(nn.Module):
    def __init__(self, in_dim, h_dim,out_dim ,n_layers,p):
        super().__init__()
        self.p = p
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.in_dim = in_dim
        self.layers_params = []

        # input layer
        self.layers_params.append(GruBlock(self.in_dim, self.h_dim,p))
        # hydden layers
        for _ in range(self.n_layers - 1):
            self.layers_params.append(GruBlock(self.h_dim, self.h_dim,p))

        for i, params in enumerate(self.layers_params):
            self.add_module(f"GRUblock_{i}", params)


    def forward(self, input: Tensor, current_hidden_state: Tensor):
        hidden_states = []
        h_t,h_t_dropout = self.layers_params[0](input, current_hidden_state[0])
        hidden_states.append(h_t)
        for i,layer in enumerate(self.layers_params[1:]):
            h_t,h_t_dropout = layer(h_t_dropout, current_hidden_state[i + 1])
            hidden_states.append(h_t)

        return hidden_states , h_t_dropout


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        self.layer_params.append(GruColumn(self.in_dim, self.h_dim, self.out_dim, self.n_layers, dropout))
        self.add_module('GruCol', self.layer_params[0])

        self.layer_params.append(nn.Linear(self.h_dim, self.out_dim, bias=True))
        self.add_module('OutputLayer', self.layer_params[1])
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        h_dropout = []
        for i in range(seq_len):
            layer_states, h_dropout_t = self.layer_params[0](layer_input[:,i,:], layer_states)
            h_dropout.append(h_dropout_t)
        # ========================
        hidden_state = torch.stack(layer_states, dim=1)
        H_dropout = torch.stack(h_dropout, dim=1)
        layer_output = self.layer_params[1](H_dropout)
        return layer_output, hidden_state

if __name__ == '__main__':
    import unittest
    import os
    import sys
    import pathlib
    import urllib
    import shutil
    import re

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    test = unittest.TestCase()
    plt.rcParams.update({'font.size': 12})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    CORPUS_URL = 'https://github.com/cedricdeboom/character-level-rnn-datasets/raw/master/datasets/shakespeare.txt'
    DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')


    def download_corpus(out_path=DATA_DIR, url=CORPUS_URL, force=False):
        pathlib.Path(out_path).mkdir(exist_ok=True)
        out_filename = os.path.join(out_path, os.path.basename(url))

        if os.path.isfile(out_filename) and not force:
            print(f'Corpus file {out_filename} exists, skipping download.')
        else:
            print(f'Downloading {url}...')
            with urllib.request.urlopen(url) as response, open(out_filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f'Saved to {out_filename}.')
        return out_filename


    corpus_path = download_corpus()

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read()

    print(f'Corpus length: {len(corpus)} chars')
    print(corpus[7:1234])

    char_to_idx, idx_to_char = char_maps(corpus)
    print(char_to_idx)

    test.assertEqual(len(char_to_idx), len(idx_to_char))
    test.assertSequenceEqual(list(char_to_idx.keys()), list(idx_to_char.values()))
    test.assertSequenceEqual(list(char_to_idx.values()), list(idx_to_char.keys()))

    corpus, n_removed = remove_chars(corpus, ['}', '$', '_', '<', '\ufeff'])
    print(f'Removed {n_removed} chars')

    # After removing the chars, re-create the mappings
    char_to_idx, idx_to_char = char_maps(corpus)


    # Wrap the actual embedding functions for calling convenience
    def embed(text):
        return chars_to_onehot(text, char_to_idx)


    def unembed(embedding):
        return onehot_to_chars(embedding, idx_to_char)


    text_snippet = corpus[3104:3148]
    print(text_snippet)
    print(embed(text_snippet[0:3]))

    test.assertEqual(text_snippet, unembed(embed(text_snippet)))
    test.assertEqual(embed(text_snippet).dtype, torch.int8)

    # Create dataset of sequences
    seq_len = 64
    vocab_len = len(char_to_idx)

    # Create labelled samples
    samples, labels = chars_to_labelled_samples(corpus, char_to_idx, seq_len, device)
    print(f'samples shape: {samples.shape}')
    print(f'labels shape: {labels.shape}')

    # Test shapes
    num_samples = (len(corpus) - 1) // seq_len
    test.assertEqual(samples.shape, (num_samples, seq_len, vocab_len))
    test.assertEqual(labels.shape, (num_samples, seq_len))

    # Test content
    for _ in range(1000):
        # random sample
        i = np.random.randint(num_samples, size=(1,))[0]
        # Compare to corpus
        test.assertEqual(unembed(samples[i]), corpus[i * seq_len:(i + 1) * seq_len],
                         msg=f"content mismatch in sample {i}")
        # Compare to labels
        sample_text = unembed(samples[i])
        label_text = str.join('', [idx_to_char[j.item()] for j in labels[i]])
        test.assertEqual(sample_text[1:], label_text[0:-1], msg=f"label mismatch in sample {i}")





    import torch.utils.data

    # Create DataLoader returning batches of samples.
    batch_size = 32

    ds_corpus = torch.utils.data.TensorDataset(samples, labels)
    sampler_corpus = SequenceBatchSampler(ds_corpus, batch_size)
    dl_corpus = torch.utils.data.DataLoader(ds_corpus, batch_size=batch_size, sampler=sampler_corpus, shuffle=False)
    x0, y0 = next(iter(dl_corpus))


    in_dim = vocab_len
    h_dim = 256
    n_layers = 3
    model = MultilayerGRU(in_dim, h_dim, out_dim=in_dim, n_layers=n_layers)
    model = model.to(device)
    print(model)

    # Test forward pass
    y, h = model(x0.to(dtype=torch.float))
    print(f'y.shape={y.shape}')
    print(f'h.shape={h.shape}')

    test.assertEqual(y.shape, (batch_size, seq_len, vocab_len))
    test.assertEqual(h.shape, (batch_size, n_layers, h_dim))
    test.assertEqual(len(list(model.parameters())), 9 * n_layers + 2)

    for _ in range(3):
        text = generate_from_model(model, "foobar", 50, (char_to_idx, idx_to_char), T=0.5)
        print(text)
        test.assertEqual(len(text), 50)

    # Pick a tiny subset of the dataset
    subset_start, subset_end = 1001, 1005
    ds_corpus_ss = torch.utils.data.Subset(ds_corpus, range(subset_start, subset_end))
    batch_size_ss = 1
    sampler_ss = SequenceBatchSampler(ds_corpus_ss, batch_size=batch_size_ss)
    dl_corpus_ss = torch.utils.data.DataLoader(ds_corpus_ss, batch_size_ss, sampler=sampler_ss, shuffle=False)

    # Convert subset to text
    subset_text = ''
    for i in range(subset_end - subset_start):
        subset_text += unembed(ds_corpus_ss[i][0])
    print(f'Text to "memorize":\n\n{subset_text}')

    import torch.nn as nn
    import torch.optim as optim
    from hw3.training import RNNTrainer

    torch.manual_seed(42)

    lr = 0.01
    num_epochs = 500

    in_dim = vocab_len
    h_dim = 128
    n_layers = 2
    loss_fn = nn.CrossEntropyLoss()
    model = MultilayerGRU(in_dim, h_dim, out_dim=in_dim, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = RNNTrainer(model, loss_fn, optimizer, device)

    for epoch in range(num_epochs):
        epoch_result = trainer.train_epoch(dl_corpus_ss, verbose=False)

        # Every X epochs, we'll generate a sequence starting from the first char in the first sequence
        # to visualize how/if/what the model is learning.
        if epoch == 0 or (epoch + 1) % 25 == 0:
            avg_loss = np.mean(epoch_result.losses)
            accuracy = np.mean(epoch_result.accuracy)
            print(f'\nEpoch #{epoch + 1}: Avg. loss = {avg_loss:.3f}, Accuracy = {accuracy:.2f}%')

            generated_sequence = generate_from_model(model, subset_text[0],
                                                            seq_len * (subset_end - subset_start),
                                                            (char_to_idx, idx_to_char), T=0.1)

            # Stop if we've successfully memorized the small dataset.
            print(generated_sequence)
            if generated_sequence == subset_text:
                break

    # Test successful overfitting
    test.assertGreater(epoch_result.accuracy, 99)
    test.assertEqual(generated_sequence, subset_text)

