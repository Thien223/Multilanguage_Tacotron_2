from math import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Embedding
from torch.nn import functional as F
from text import sequence_to_text
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from plotting_utils import fontproperties

plot = False


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, speaker_embedding_dims, language_embedding_dims):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim + speaker_embedding_dims , attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,attention_location_kernel_size,attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        # if plot:
        #     plt.figure()
        #     plt.imshow(memory.float()[0].data.cpu().numpy())
        #     plt.savefig('attention_encoder_output.png')
        #     print(f'model.py 76 encoder output shape: {memory.shape}')
        # if plot:
        #     plt.figure()
        #     plt.imshow(memory.float()[0].data.cpu().numpy())
        #     plt.savefig('attention_encoder_output.png')
        #     print(f'model.py 76 encoder output shape: {memory.shape}')
        # if plot:
        #     plt.figure()
        #     plt.imshow(memory.float()[0].data.cpu().numpy())
        #     plt.savefig('attention_encoder_output.png')
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class Conv1dGenerated(torch.nn.Module):
    """One dimensional convolution with generated weights (each language has separate weights).

    Arguments:
        hparams -- params to construct models
    """

    # def __init__(self, embedding_dim, bottleneck_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    def __init__(self, hparams, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(Conv1dGenerated, self).__init__()

        self._in_channels = hparams.symbols_embedding_dim * hparams.languages_number
        self._out_channels = hparams.symbols_embedding_dim * hparams.languages_number #### todo: redefine hparams.encoder_embedding_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = hparams.languages_number

        # in_channels and out_channels is divisible by groups
        # tf.nn.functional.conv1d accepts weights of shape [out_channels, in_channels // groups, kernel]
        self._bottleneck = Linear(hparams.generator_dim, hparams.bottleneck_dim)
        #### group is number of language
        #### outchannel is encoder output dimension
        ### kernel size = 1
        ### inchanel  = input dim of each input (text embedding dim) * number of language (group)
        kernel_output_size =  (self._out_channels // self._groups )* (self._in_channels // self._groups) * self._kernel_size
        bias_output_size =  self._out_channels // self._groups
        self._kernel = Linear(hparams.bottleneck_dim, kernel_output_size)
        self._bias = Linear(hparams.bottleneck_dim, bias_output_size) if bias else None

    def forward(self, inputs_):
        language_embedding, inputs = inputs_
        assert language_embedding.shape[0] == self._groups, ('Number of groups of a convolutional layer must match the number of generators.')
        language_embeded = self._bottleneck(language_embedding)
        kernel = self._kernel(language_embeded).view(self._out_channels, self._in_channels // self._groups, self._kernel_size)
        bias = self._bias(language_embeded).view(self._out_channels) if self._bias else None
        output = F.conv1d(inputs, kernel, bias, self._stride, self._padding, self._dilation, self._groups)
        return language_embedding, output


class BatchNorm1dGenerated(torch.nn.Module):
    """One dimensional batch normalization with generated weights (each group has separate parameters).

    Arguments:
        embedding_dim -- size of the meta embedding (should be language embedding)
        bottleneck_dim -- size of the generating embedding
        see torch.nn.BatchNorm1d
    Keyword arguments:
        groups -- number of groups with separate weights
    """

    def __init__(self, hparams, eps=1e-8, momentum=0.1):
        super(BatchNorm1dGenerated, self).__init__()

        num_features = hparams.symbols_embedding_dim * hparams.languages_number
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self._num_features = num_features // hparams.languages_number
        self._eps = eps
        self._momentum = momentum
        self._groups = hparams.languages_number
        ### bottleneck dim is 2
        self._bottleneck = Linear(hparams.generator_dim, hparams.bottleneck_dim)
        ### number_features is encoder's output dimension (512)
        self._affine = Linear(hparams.bottleneck_dim, self._num_features + self._num_features)

    def forward(self, inputs_):
        language_embedding, inputs = inputs_
        assert language_embedding.shape[0] == self._groups, (
            'Number of groups of a batchnorm layer must match the number of generators.')
        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum
        e = self._bottleneck(language_embedding)
        affine = self._affine(e)
        scale = affine[:, :self._num_features].contiguous().view(-1)
        bias = affine[:, self._num_features:].contiguous().view(-1)
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self._momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self._momentum
        output = F.batch_norm(inputs, self.running_mean, self.running_var, scale, bias, self.training, exponential_average_factor, self._eps)
        return language_embedding, output


# noinspection PyArgumentList
class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.conv_layer = Conv1dGenerated(hparams=hparams,kernel_size=hparams.encoder_kernel_size,stride=1,padding=int((hparams.encoder_kernel_size - 1) / 2), bias=False)
        self.batchnorm_layer = BatchNorm1dGenerated(hparams)
        self.hparams = hparams
        self.lstm = nn.LSTM(hparams.symbols_embedding_dim,int(hparams.encoder_embedding_dim / 2), 1,batch_first=True, bidirectional=True)
        self.dnn = LinearNorm(hparams.symbols_embedding_dim * hparams.languages_number, hparams.encoder_embedding_dim * hparams.languages_number)
        self.language_embedding_layer = Embedding(self.hparams.languages_number, self.hparams.generator_dim)
        convolutions = []
        for _ in range(self.hparams.encoder_n_convolutions):
            conv = nn.Sequential(
                Conv1dGenerated(hparams=hparams, kernel_size=hparams.encoder_kernel_size, stride=1, padding=int((hparams.encoder_kernel_size - 1) / 2), bias=False),
                BatchNorm1dGenerated(hparams))
            self.convolutions = convolutions.append(conv)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, inputs, input_lengths, language_ids):
        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('train_embedded_inputs_text.png')
        language_embedding = self.language_embedding_layer(torch.arange(start=0, end=self.hparams.languages_number, device=inputs.device))

        input_dim = inputs.shape[1]
        inputs = inputs.reshape(self.hparams.batch_size // self.hparams.languages_number, self.hparams.languages_number * input_dim, -1)

        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('train_convolution_inputs.png')
        for conv in self.convolutions:
            _, inputs = conv((language_embedding, inputs))
            inputs = F.dropout(F.relu(inputs), 0.5, self.training)
        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('train_convolution_outputs.png')
        inputs = inputs.reshape(self.hparams.batch_size, self.hparams.encoder_embedding_dim, -1).transpose(1,2)

        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('train_convolution_outputs_reshaped.png')
        # inputs = self.dnn(inputs)
        # plt.imshow(inputs.float()[0].data.cpu().numpy())
        # plt.savefig('training_encoder_inputs.png')

        self.lstm.flatten_parameters()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        outputs, _ = self.lstm(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if plot:
            plt.figure()
            plt.imshow(outputs.float()[0].data.cpu().numpy())
            plt.savefig('training_encoder_outputs.png')
        plt.close('all')
        return outputs

    def inference(self, inputs, language_ids):
        if self.hparams.fp16_run:
            language_ids = language_ids.half()
        inputs = inputs.expand((self.hparams.languages_number, -1, -1))
        language_embedding = self.language_embedding_layer(torch.arange(self.hparams.languages_number, device=inputs.device))
        batch_size = inputs.shape[0]
        input_dim = inputs.shape[1]
        output_dim = input_dim  ### todo: until now, we use encoder's output dim == input dim = 512. config this line to get encoder's output dim from hparams if they are different



        inputs = inputs.reshape(batch_size // self.hparams.languages_number, self.hparams.languages_number * input_dim, -1)
        print(inputs.shape)
        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('inference_convolution_inputs.png')

        for conv in self.convolutions:
            _, inputs = conv((language_embedding, inputs))
            inputs = F.dropout(F.relu(inputs), 0.5, self.training)

        if plot:
            plt.figure()
            plt.imshow(inputs.float()[0].data.cpu().numpy())
            plt.savefig('inference_convolution_outputs.png')


        inputs = inputs.transpose(1, 2)
        # inputs = self.dnn(inputs)
        inputs = inputs.reshape(batch_size, self.hparams.encoder_embedding_dim, -1).transpose(1, 2)
        self.lstm.flatten_parameters()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        outputs, _ = self.lstm(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # plt.figure()
        # plt.imshow(inputs.float()[0].data.cpu().numpy())
        # plt.savefig('inferencing_encoder_inputs.png')

        # ######
        # plt.figure()
        # plt.imshow(inputs.float()[0].data.cpu().numpy())
        # plt.savefig('inference_lstm_outputs.png')
        #####
        temp_outputs = torch.zeros(1, inputs.shape[1], inputs.shape[2], device=inputs.device, dtype=inputs.dtype)
        inputs_lang_norm = language_ids / language_ids.sum(dim=2, keepdim=True)[0]

        for language_id in range(self.hparams.languages_number):
            lang_weight = inputs_lang_norm[0,:,language_id].reshape(-1, 1)
            temp_outputs[0] = temp_outputs[0] + lang_weight * inputs[language_id]
        outputs = temp_outputs

        if plot:
            plt.figure()
            plt.imshow(outputs.float()[0].data.cpu().numpy())
            plt.savefig('inferencing_encoder_outputs.png')
        plt.close('all')
        return outputs


# noinspection PyArgumentList
class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.speaker_embedding = self.get_embedding(hparams.speaker_embedding_dims, hparams.speakers_number)
        self.language_embedding = self.get_embedding(hparams.lang_embedding_dims, hparams.languages_number)
        self.speaker_embedding_dims = hparams.speaker_embedding_dims
        self.lang_embedding_dims = hparams.lang_embedding_dims

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims, hparams.attention_rnn_dim)

        self.attention_layer = Attention(hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size, hparams.speaker_embedding_dims, hparams.lang_embedding_dims)

        self.decoder_rnn = nn.LSTMCell(hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims, 1,
            bias=True, w_init_gain='sigmoid')

    def get_embedding(self, embedding_dimension, size=None):
        embedding = Embedding(size, embedding_dimension)
        torch.nn.init.xavier_uniform_(embedding.weight)
        return embedding

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask, speaker_dim, lang_dim):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim+speaker_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, step=0):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))


        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)


        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory,attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, speaker_ids, language_ids):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        embedded_speaker = self.speaker_embedding(speaker_ids)
        memory = torch.cat((memory, embedded_speaker), dim=-1)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths), speaker_dim=self.speaker_embedding_dims, lang_dim=self.lang_embedding_dims)
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory,speaker_ids, language_ids):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # if language_ids is not None and language_ids.dim() == 3:
        #     language_ids = torch.argmax(language_ids, dim=2) # convert one-hot into indices
        ### calculate embedded language and speaker, then concat to encoder ouput before decoding
        embedded_speaker = self.speaker_embedding(speaker_ids)
        # embedded_language = self.language_embedding(language_ids)
        memory = torch.cat((memory, embedded_speaker), dim=-1)
        # memory = torch.cat((memory, embedded_language), dim=-1)
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None, speaker_dim=self.speaker_embedding_dims, lang_dim=self.lang_embedding_dims)
        mel_outputs, gate_outputs, alignments = [], [], []
        step=0
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input, step=step)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]
            step +=1

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        print(f'alignments shape {alignments.shape}')
        if plot:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(alignments[0].float().data.cpu().numpy().T, aspect='auto', origin='lower', interpolation='none')
            fig.colorbar(im, ax=ax)
            xlabel = 'Decoder timestep'
            plt.xlabel(xlabel, fontproperties=fontproperties)
            plt.ylabel('Encoder timestep', fontproperties=fontproperties)
            plt.tight_layout()
            plt.savefig(f'inference_attention_alignment.png')
            plt.close()
        return mel_outputs, gate_outputs, alignments



class Speaker_Embedding(nn.Module):
    """Disciminative Embedding module:
        Simple embedding layer with init weight
    """

    def __init__(self, hparams):
        super(Speaker_Embedding, self).__init__()
        self.speaker_embedding = nn.Embedding(hparams.speakers_number, embedding_dim=hparams.speaker_embedding_dims)  ### hparams.speakers_number is number of speakers
        torch.nn.init.xavier_uniform_(self.speaker_embedding.weight)
    def forward(self, x):
        ### x should has [batch, 2] shape because there are 2 speakers
        embedded = self.speaker_embedding(x)
        return embedded ### final_outputs.shape[0] = batch size


class Language_Embedding(nn.Module):
    """Language  Embedding module:
        Simple embedding layer with init weight
    """

    def __init__(self, hparams):
        super(Language_Embedding, self).__init__()
        self.language_embedding = nn.Embedding(hparams.languages_number, embedding_dim=hparams.lang_embedding_dims)  ### hparams.languages_number is number of languages. 32 is embedding dimension
        torch.nn.init.xavier_uniform_(self.language_embedding.weight)
    def forward(self, x):
        ### x should has [batch, 2] shape because there are 2 languages
        embedded = self.language_embedding(x)
        return embedded ### final_outputs.shape[0] = batch size


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.speakers_number = hparams.speakers_number
        self.languages_number = hparams.languages_number

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        ### format variables' datatype and send them to GPU
        languages_ids, speaker_ids, text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        languages_ids = to_gpu(languages_ids).long()
        speaker_ids = to_gpu(speaker_ids).long()
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        return ((languages_ids, speaker_ids,text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        language_ids, speaker_ids, text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        #
        # language_ids = language_ids.unsqueeze(-1).expand((language_ids.shape[0], text_inputs.shape[1], self.languages_number))  ### shape: [batch, max_length, language_number]
        # one_hots = torch.zeros(language_ids.size(0), language_ids.size(1), language_ids.size(2)).zero_().to(text_inputs.device)  ### zeros's tensor with shape: [batch, max_length, language_number]
        # language_ids = one_hots.scatter_(dim=2, index=language_ids, value=1)  ### shape: [batch, max_length, language_number]
        #

        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        #### todo: with this project. we use only 1 speaker for each language. then language id and speaker id can be used exchangeably.
        # todo: config this line (and data_utils\TextMelCollate\__call__ function to return language id if there are more than 1 speaker per language.
        ### until now, speaker_ids has [batch, max_length] shape

        #### ENCODING
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths,language_ids)

        # plt.figure()
        # plt.imshow(encoder_outputs.float()[0].data.cpu().numpy())
        # plt.savefig('training_encoder_outputs.png')
        #### DECODING
        #
        # plt.figure()
        # plt.imshow(mels.float()[0].data.cpu().numpy())
        # plt.savefig('target_mels.png')

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths, speaker_ids=speaker_ids, language_ids=language_ids)

        # plt.figure()
        # plt.imshow(mel_outputs.float()[0].data.cpu().numpy())
        # plt.savefig('decoder_mel_outputs.png')

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        output = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],output_lengths)
        #

        padded_sequences = text_inputs.cpu().numpy()
        seqs = [[v for v in padded_sequence if v != 0] for padded_sequence in padded_sequences]
        txts = [sequence_to_text(txt) for txt in seqs]


        if plot:
            plt.figure()
            plt.imshow(output[0].float()[0].data.cpu().numpy())
            plt.xlabel(txts[0],fontproperties=fontproperties)
            plt.savefig('train_final_outputs.png')

            plt.figure()
            plt.imshow(mels[0].float().data.cpu().numpy())
            plt.xlabel(txts[0],fontproperties=fontproperties)
            plt.savefig('train_target_outputs.png')

            plt.figure()
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(alignments[0].float().data.cpu().numpy().T, aspect='auto', origin='lower', interpolation='none')
            fig.colorbar(im, ax=ax)
            xlabel = 'Decoder timestep'
            plt.xlabel(xlabel, fontproperties=fontproperties)
            plt.ylabel('Encoder timestep', fontproperties=fontproperties)
            plt.tight_layout()
            plt.title(txts[0],fontproperties=fontproperties)
            plt.savefig(f'train_attention_alignment.png')
            plt.close('all')
        return output

    def inference(self, inputs, speaker_ids, language_ids):
        ### for our project, with each language, we have 1 speaker then speaker and language can be used as the same
        #### expand language_id to [batch, max_length, language_number] shape
        language_ids = language_ids.unsqueeze(-1).expand((language_ids.shape[0], inputs.shape[1], self.languages_number))  ### shape: [batch, max_length, language_number]
        one_hots = torch.zeros(language_ids.size(0), language_ids.size(1), language_ids.size(2)).zero_()  ### zeros's tensor with shape: [batch, max_length, language_number]
        language_ids = one_hots.scatter_(dim=2, index=language_ids, value=1).to(inputs.device) ### shape: [batch, max_length, language_number]
        speaker_ids = speaker_ids.to(inputs.device)

        # print(f'language_ids 2 {language_ids}')

        #### ENCODING
        print(f'input {inputs}')
        print(f'input shape {inputs.shape}')
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        if plot:
            plt.figure()
            plt.imshow(embedded_inputs.float()[0].data.cpu().numpy())
            plt.savefig('embedded_inputs_text.png')
        encoder_outputs = self.encoder.inference(embedded_inputs, language_ids=language_ids)
        # plt.figure()
        # plt.imshow(encoder_outputs.float()[0].data.cpu().numpy())
        # plt.savefig('inference_encoder_outputs.png')

        ### DECODING
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs, speaker_ids=speaker_ids, language_ids=language_ids)
        # plt.figure()
        # plt.imshow(mel_outputs.float()[0].data.cpu().numpy())
        # plt.savefig('inference_mel_outputs.png')


        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        #
        if plot:
            plt.figure()
            plt.imshow(outputs[0].float()[0].data.cpu().numpy())
            plt.savefig('inference_final_outputs.png')

        return outputs
### backup
#
# from math import sqrt
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pylab as plt
# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.nn import Linear
# from torch.nn import functional as F
#
# from layers import ConvNorm, LinearNorm
# from utils import to_gpu, get_mask_from_lengths
#
#
# class LocationLayer(nn.Module):
#     def __init__(self, attention_n_filters, attention_kernel_size,
#                  attention_dim):
#         super(LocationLayer, self).__init__()
#         padding = int((attention_kernel_size - 1) / 2)
#         self.location_conv = ConvNorm(2, attention_n_filters,
#                                       kernel_size=attention_kernel_size,
#                                       padding=padding, bias=False, stride=1,
#                                       dilation=1)
#         self.location_dense = LinearNorm(attention_n_filters, attention_dim,
#                                          bias=False, w_init_gain='tanh')
#
#     def forward(self, attention_weights_cat):
#         processed_attention = self.location_conv(attention_weights_cat)
#         processed_attention = processed_attention.transpose(1, 2)
#         processed_attention = self.location_dense(processed_attention)
#         return processed_attention
#
#
# class Attention(nn.Module):
#     def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
#                  attention_location_n_filters, attention_location_kernel_size, speaker_embedding_dims, language_embedding_dims):
#         super(Attention, self).__init__()
#         self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
#                                       bias=False, w_init_gain='tanh')
#         self.memory_layer = LinearNorm(embedding_dim + speaker_embedding_dims + language_embedding_dims, attention_dim, bias=False,
#                                        w_init_gain='tanh')
#         self.v = LinearNorm(attention_dim, 1, bias=False)
#         self.location_layer = LocationLayer(attention_location_n_filters,attention_location_kernel_size,attention_dim)
#         self.score_mask_value = -float("inf")
#
#     def get_alignment_energies(self, query, processed_memory,
#                                attention_weights_cat):
#         """
#         PARAMS
#         ------
#         query: decoder output (batch, n_mel_channels * n_frames_per_step)
#         processed_memory: processed encoder outputs (B, T_in, attention_dim)
#         attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
#
#         RETURNS
#         -------
#         alignment (batch, max_time)
#         """
#
#         processed_query = self.query_layer(query.unsqueeze(1))
#         processed_attention_weights = self.location_layer(attention_weights_cat)
#         energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
#         energies = energies.squeeze(-1)
#         return energies
#
#     def forward(self, attention_hidden_state, memory, processed_memory,attention_weights_cat, mask):
#         """
#         PARAMS
#         ------
#         attention_hidden_state: attention rnn last output
#         memory: encoder outputs
#         processed_memory: processed encoder outputs
#         attention_weights_cat: previous and cummulative attention weights
#         mask: binary mask for padded data
#         """
#         alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)
#
#         if mask is not None:
#             alignment.data.masked_fill_(mask, self.score_mask_value)
#
#         attention_weights = F.softmax(alignment, dim=1)
#         attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
#         attention_context = attention_context.squeeze(1)
#         return attention_context, attention_weights
#
#
# class Prenet(nn.Module):
#     def __init__(self, in_dim, sizes):
#         super(Prenet, self).__init__()
#         in_sizes = [in_dim] + sizes[:-1]
#         self.layers = nn.ModuleList(
#             [LinearNorm(in_size, out_size, bias=False)
#              for (in_size, out_size) in zip(in_sizes, sizes)])
#
#     def forward(self, x):
#         for linear in self.layers:
#             x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
#         return x
#
#
# class Postnet(nn.Module):
#     """Postnet
#         - Five 1-d convolution with 512 channels and kernel size 5
#     """
#
#     def __init__(self, hparams):
#         super(Postnet, self).__init__()
#         self.convolutions = nn.ModuleList()
#
#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='tanh'),
#                 nn.BatchNorm1d(hparams.postnet_embedding_dim))
#         )
#
#         for i in range(1, hparams.postnet_n_convolutions - 1):
#             self.convolutions.append(
#                 nn.Sequential(
#                     ConvNorm(hparams.postnet_embedding_dim,
#                              hparams.postnet_embedding_dim,
#                              kernel_size=hparams.postnet_kernel_size, stride=1,
#                              padding=int((hparams.postnet_kernel_size - 1) / 2),
#                              dilation=1, w_init_gain='tanh'),
#                     nn.BatchNorm1d(hparams.postnet_embedding_dim))
#             )
#
#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='linear'),
#                 nn.BatchNorm1d(hparams.n_mel_channels))
#             )
#
#     def forward(self, x):
#         for i in range(len(self.convolutions) - 1):
#             x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
#         x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
#         return x
#
#
# class Conv1dGenerated(torch.nn.Module):
#     """One dimensional convolution with generated weights (each language has separate weights).
#
#     Arguments:
#         hparams -- params to construct models
#     """
#
#     # def __init__(self, embedding_dim, bottleneck_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#     def __init__(self, hparams, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
#         super(Conv1dGenerated, self).__init__()
#
#         self._in_channels = hparams.encoder_embedding_dim * hparams.languages_number
#         self._out_channels = hparams.encoder_embedding_dim * hparams.languages_number #### todo: redefine hparams.encoder_embedding_dim
#         self._kernel_size = kernel_size
#         self._stride = stride
#         self._padding = padding
#         self._dilation = dilation
#         self._groups = hparams.languages_number
#         self.lang_embedding_dims = hparams.lang_embedding_dims
#         self.bottleneck_dim = hparams.bottleneck_dim
#
#         # in_channels and out_channels is divisible by groups
#         # tf.nn.functional.conv1d accepts weights of shape [out_channels, in_channels // groups, kernel]
#         self._bottleneck = Linear(self.lang_embedding_dims, self.bottleneck_dim)
#         #### group is number of language
#         #### outchannel is encoder output dimension
#         ### kernel size = 1
#         ### inchanel  = input dim of each input (text embedding dim) * number of language (group)
#         kernel_output_size =  self._out_channels // self._groups * self._in_channels // self._groups * self._kernel_size
#         bias_output_size =  self._out_channels // self._groups
#         self._kernel = Linear(self.bottleneck_dim, kernel_output_size)
#         self._bias = Linear(self.bottleneck_dim, bias_output_size) if bias else None
#
#
#     def forward(self, language_embedding, inputs):
#         assert language_embedding.shape[0] == self._groups, ('Number of groups of a convolutional layer must match the number of generators.')
#         language_embedding = self._bottleneck(language_embedding)
#         kernel = self._kernel(language_embedding).view(self._out_channels, self._in_channels // self._groups, self._kernel_size)
#         bias = self._bias(language_embedding).view(self._out_channels) if self._bias else None
#         output = F.conv1d(inputs, kernel, bias, self._stride, self._padding, self._dilation, self._groups)
#         return output
#
#
# class BatchNorm1dGenerated(torch.nn.Module):
#     """One dimensional batch normalization with generated weights (each group has separate parameters).
#
#     Arguments:
#         embedding_dim -- size of the meta embedding (should be language embedding)
#         bottleneck_dim -- size of the generating embedding
#         see torch.nn.BatchNorm1d
#     Keyword arguments:
#         groups -- number of groups with separate weights
#     """
#
#     def __init__(self, hparams, eps=1e-8, momentum=0.1):
#         super(BatchNorm1dGenerated, self).__init__()
#
#         num_features = hparams.encoder_embedding_dim * hparams.languages_number
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))
#         self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         self._num_features = num_features // hparams.languages_number
#         self._eps = eps
#         self._momentum = momentum
#         self._groups = hparams.languages_number
#         ### bottleneck dim is 8
#         self._bottleneck = Linear(hparams.lang_embedding_dims, hparams.bottleneck_dim)
#         ### number_features is encoder's output dimension (512)
#         self._affine = Linear(hparams.bottleneck_dim, self._num_features + self._num_features)
#
#     def forward(self, language_embedding, inputs):
#         assert language_embedding.shape[0] == self._groups, (
#             'Number of groups of a batchnorm layer must match the number of generators.')
#         if self._momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self._momentum
#         e = self._bottleneck(language_embedding)
#         affine = self._affine(e)
#         scale = affine[:, :self._num_features].contiguous().view(-1)
#         bias = affine[:, self._num_features:].contiguous().view(-1)
#         if self.training:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self._momentum is None:
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:
#                     exponential_average_factor = self._momentum
#         output = F.batch_norm(inputs, self.running_mean, self.running_var, scale, bias, self.training, exponential_average_factor, self._eps)
#         return output
#
#
# # noinspection PyArgumentList
# class Encoder(nn.Module):
#     """Encoder module:
#         - Three 1-d convolution banks
#         - Bidirectional LSTM
#     """
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#
#
#
#         self.conv_layer = Conv1dGenerated(hparams=hparams,kernel_size=hparams.encoder_kernel_size,stride=1,padding=int((hparams.encoder_kernel_size - 1) / 2))
#
#         self.batchnorm_layer = BatchNorm1dGenerated(hparams)
#
#         self.hparams = hparams
#         self.lstm = nn.LSTM(hparams.encoder_embedding_dim,int(hparams.encoder_embedding_dim / 2), 1,batch_first=True, bidirectional=True)
#         self.language_embedding_layer = Language_Embedding(hparams)
#
#     def forward(self, inputs, input_lengths):
#         language_embedding = self.language_embedding_layer(torch.arange(self.hparams.languages_number, device=inputs.device))
#         input_dim = inputs.shape[1]
#         output_dim = input_dim ### todo: until now, we use encoder's output dim == input dim = 512
#
#         inputs = inputs.reshape(self.hparams.batch_size // self.hparams.languages_number, self.hparams.languages_number * input_dim, -1)
#
#
#         for _ in range(self.hparams.encoder_n_convolutions):
#             conv_output = self.conv_layer(language_embedding, inputs)
#             batchnorm_output = self.batchnorm_layer(language_embedding, conv_output)
#             relu_output = F.relu(batchnorm_output)
#             inputs = F.dropout(relu_output, 0.5, self.training)
#
#         inputs = inputs.reshape(self.hparams.batch_size, output_dim, -1)
#         # plt.imshow(inputs.float()[0].data.cpu().numpy())
#         # plt.savefig('training_encoder_inputs.png')
#         x = inputs.transpose(1, 2)
#
#         # pytorch tensor are not reversible, hence the conversion
#         input_lengths = input_lengths.cpu().numpy()
#         x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
#
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#         return outputs
#
#     def inference(self, inputs, language_ids):
#         if self.hparams.fp16_run:
#             language_ids = language_ids.half()
#         #### at inferencing time, we pass language ids as list
#         # language_ids = torch.FloatTensor(language_ids).to(inputs.device)
#         # language_ids = language_ids.unsqueeze(-1).expand(language_ids.shape[0], inputs.shape[1])
#         # language_ids = language_ids.unsqueeze(-1).expand((language_ids.shape[0], inputs.shape[1], self.hparams.languages_number)) ### language_ids should has shape: [batch, max_input_length, language_number)
#         ### because input passed as 1 language, expand to fit language number dim.
#         inputs = inputs.expand((self.hparams.languages_number, -1, -1))
#
#         language_embedding = self.language_embedding_layer(torch.arange(self.hparams.languages_number, device=inputs.device))
#         batch_size = inputs.shape[0]
#         input_dim = inputs.shape[1]
#         output_dim = input_dim  ### todo: until now, we use encoder's output dim == input dim = 512. config this line to get encoder's output dim from hparams if they are different
#         inputs = inputs.reshape(batch_size // self.hparams.languages_number, self.hparams.languages_number * input_dim, -1)
#
#         for _ in range(self.hparams.encoder_n_convolutions):
#             conv_output = self.conv_layer(language_embedding, inputs)
#             batchnorm_output = self.batchnorm_layer(language_embedding, conv_output)
#             relu_output = F.relu(batchnorm_output)
#             inputs = F.dropout(relu_output, 0.5, self.training)
#
#
#         inputs = inputs.reshape(batch_size, output_dim, -1)
#         inputs = inputs.transpose(1, 2)
#
#         self.lstm.flatten_parameters()
#         inputs, _ = self.lstm(inputs)
#         #
#         # plt.figure()
#         # plt.imshow(inputs.float()[0].data.cpu().numpy())
#         # plt.savefig('inferencing_encoder_inputs.png')
#
#
#         #####
#         temp_outputs = torch.zeros(1, inputs.shape[1], inputs.shape[2], device=inputs.device, dtype=inputs.dtype)
#         inputs_lang_norm = language_ids / language_ids.sum(dim=2, keepdim=True)[0]
#
#         for language_id in range(self.hparams.languages_number):
#             lang_weight = inputs_lang_norm[0,:,language_id].reshape(-1, 1)
#             temp_outputs[0] = temp_outputs[0] + lang_weight * inputs[language_id]
#         outputs = temp_outputs
#
#
#         ######
#         plt.figure()
#         plt.imshow(outputs.float()[0].data.cpu().numpy())
#         plt.savefig('inferencing_encoder_outputs.png')
#         return outputs
#
#
# # noinspection PyArgumentList
# class Decoder(nn.Module):
#     def __init__(self, hparams):
#         super(Decoder, self).__init__()
#         self.n_mel_channels = hparams.n_mel_channels
#         self.n_frames_per_step = hparams.n_frames_per_step
#         self.encoder_embedding_dim = hparams.encoder_embedding_dim
#         self.attention_rnn_dim = hparams.attention_rnn_dim
#         self.decoder_rnn_dim = hparams.decoder_rnn_dim
#         self.prenet_dim = hparams.prenet_dim
#         self.max_decoder_steps = hparams.max_decoder_steps
#         self.gate_threshold = hparams.gate_threshold
#         self.p_attention_dropout = hparams.p_attention_dropout
#         self.p_decoder_dropout = hparams.p_decoder_dropout
#         self.speaker_embedding = Speaker_Embedding(hparams)
#         self.language_embedding = Language_Embedding(hparams)
#         self.speaker_embedding_dims = hparams.speaker_embedding_dims
#         self.lang_embedding_dims = hparams.lang_embedding_dims
#
#         self.prenet = Prenet(
#             hparams.n_mel_channels * hparams.n_frames_per_step,
#             [hparams.prenet_dim, hparams.prenet_dim])
#
#         self.attention_rnn = nn.LSTMCell(
#             hparams.prenet_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims+hparams.lang_embedding_dims, hparams.attention_rnn_dim)
#
#         self.attention_layer = Attention(hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
#             hparams.attention_dim, hparams.attention_location_n_filters,
#             hparams.attention_location_kernel_size, hparams.speaker_embedding_dims, hparams.lang_embedding_dims)
#
#         self.decoder_rnn = nn.LSTMCell(hparams.attention_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims + hparams.lang_embedding_dims,
#             hparams.decoder_rnn_dim, 1)
#
#         self.linear_projection = LinearNorm(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims + hparams.lang_embedding_dims,
#             hparams.n_mel_channels * hparams.n_frames_per_step)
#
#         self.gate_layer = LinearNorm(hparams.decoder_rnn_dim + hparams.encoder_embedding_dim + hparams.speaker_embedding_dims + hparams.lang_embedding_dims, 1,
#             bias=True, w_init_gain='sigmoid')
#
#     def get_go_frame(self, memory):
#         """ Gets all zeros frames to use as first decoder input
#         PARAMS
#         ------
#         memory: decoder outputs
#
#         RETURNS
#         -------
#         decoder_input: all zeros frames
#         """
#         B = memory.size(0)
#         decoder_input = Variable(memory.data.new(
#             B, self.n_mel_channels * self.n_frames_per_step).zero_())
#         return decoder_input
#
#     def initialize_decoder_states(self, memory, mask, speaker_dim, lang_dim):
#         """ Initializes attention rnn states, decoder rnn states, attention
#         weights, attention cumulative weights, attention context, stores memory
#         and stores processed memory
#         PARAMS
#         ------
#         memory: Encoder outputs
#         mask: Mask for padded data if training, expects None for inference
#         """
#         B = memory.size(0)
#         MAX_TIME = memory.size(1)
#
#         self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
#         self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
#
#         self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
#         self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
#
#         self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
#         self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
#         self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim+speaker_dim+lang_dim).zero_())
#
#         self.memory = memory
#         self.processed_memory = self.attention_layer.memory_layer(memory)
#         self.mask = mask
#
#     def parse_decoder_inputs(self, decoder_inputs):
#         """ Prepares decoder inputs, i.e. mel outputs
#         PARAMS
#         ------
#         decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
#
#         RETURNS
#         -------
#         inputs: processed decoder inputs
#
#         """
#         # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
#         decoder_inputs = decoder_inputs.transpose(1, 2)
#         decoder_inputs = decoder_inputs.view(
#             decoder_inputs.size(0),
#             int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
#         # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
#         decoder_inputs = decoder_inputs.transpose(0, 1)
#         return decoder_inputs
#
#     def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
#         """ Prepares decoder outputs for output
#         PARAMS
#         ------
#         mel_outputs:
#         gate_outputs: gate output energies
#         alignments:
#
#         RETURNS
#         -------
#         mel_outputs:
#         gate_outpust: gate output energies
#         alignments:
#         """
#         # (T_out, B) -> (B, T_out)
#         alignments = torch.stack(alignments).transpose(0, 1)
#         # (T_out, B) -> (B, T_out)
#         gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
#         gate_outputs = gate_outputs.contiguous()
#         # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
#         mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
#         # decouple frames per step
#         mel_outputs = mel_outputs.view(
#             mel_outputs.size(0), -1, self.n_mel_channels)
#         # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
#         mel_outputs = mel_outputs.transpose(1, 2)
#
#         return mel_outputs, gate_outputs, alignments
#
#     def decode(self, decoder_input):
#         """ Decoder step using stored states, attention and memory
#         PARAMS
#         ------
#         decoder_input: previous mel output
#         RETURNS
#         -------
#         mel_output:
#         gate_output: gate output energies
#         attention_weights:
#         """
#
#         cell_input = torch.cat((decoder_input, self.attention_context), -1)
#
#         self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
#
#         self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)
#
#         attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)
#
#
#         self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory,attention_weights_cat, self.mask)
#
#         self.attention_weights_cum += self.attention_weights
#
#         decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
#
#         self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
#         self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
#
#         decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
#
#         decoder_output = self.linear_projection(decoder_hidden_attention_context)
#
#         gate_prediction = self.gate_layer(decoder_hidden_attention_context)
#         return decoder_output, gate_prediction, self.attention_weights
#
#     def forward(self, memory, decoder_inputs, memory_lengths, speaker_ids, language_ids):
#         """ Decoder forward pass for training
#         PARAMS
#         ------
#         memory: Encoder outputs
#         decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
#         memory_lengths: Encoder output lengths for attention masking.
#
#         RETURNS
#         -------
#         mel_outputs: mel outputs from the decoder
#         gate_outputs: gate outputs from the decoder
#         alignments: sequence of attention weights from the decoder
#         """
#         if language_ids is not None and language_ids.dim() == 3:
#             language_ids = torch.argmax(language_ids, dim=2) # convert one-hot into indices
#         ### calculate embedded language and speaker, then concat to encoder ouput before decoding
#         embedded_speaker = self.speaker_embedding(speaker_ids)
#         embedded_language = self.language_embedding(language_ids)
#         memory = torch.cat((memory, embedded_speaker), dim=-1)
#         memory = torch.cat((memory, embedded_language), dim=-1)
#         decoder_input = self.get_go_frame(memory).unsqueeze(0)
#         decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
#         decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
#         decoder_inputs = self.prenet(decoder_inputs)
#         self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths), speaker_dim=self.speaker_embedding_dims, lang_dim=self.lang_embedding_dims)
#         mel_outputs, gate_outputs, alignments = [], [], []
#         while len(mel_outputs) < decoder_inputs.size(0) - 1:
#             decoder_input = decoder_inputs[len(mel_outputs)]
#             mel_output, gate_output, attention_weights = self.decode(decoder_input)
#             mel_outputs += [mel_output.squeeze(1)]
#             gate_outputs += [gate_output.squeeze(1)]
#             alignments += [attention_weights]
#         mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
#         return mel_outputs, gate_outputs, alignments
#
#     def inference(self, memory,speaker_ids, language_ids):
#         """ Decoder inference
#         PARAMS
#         ------
#         memory: Encoder outputs
#
#         RETURNS
#         -------
#         mel_outputs: mel outputs from the decoder
#         gate_outputs: gate outputs from the decoder
#         alignments: sequence of attention weights from the decoder
#         """
#         if language_ids is not None and language_ids.dim() == 3:
#             language_ids = torch.argmax(language_ids, dim=2) # convert one-hot into indices
#         ### calculate embedded language and speaker, then concat to encoder ouput before decoding
#         embedded_speaker = self.speaker_embedding(speaker_ids)
#         embedded_language = self.language_embedding(language_ids)
#         memory = torch.cat((memory, embedded_speaker), dim=-1)
#         memory = torch.cat((memory, embedded_language), dim=-1)
#         decoder_input = self.get_go_frame(memory)
#         self.initialize_decoder_states(memory, mask=None, speaker_dim=self.speaker_embedding_dims, lang_dim=self.lang_embedding_dims)
#         mel_outputs, gate_outputs, alignments = [], [], []
#         while True:
#             decoder_input = self.prenet(decoder_input)
#             mel_output, gate_output, alignment = self.decode(decoder_input)
#             mel_outputs += [mel_output.squeeze(1)]
#             gate_outputs += [gate_output]
#             alignments += [alignment]
#             if torch.sigmoid(gate_output.data) > self.gate_threshold:
#                 break
#             elif len(mel_outputs) == self.max_decoder_steps:
#                 print("Warning! Reached max decoder steps")
#                 break
#             decoder_input = mel_output
#         mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
#         return mel_outputs, gate_outputs, alignments
#
#
#
# class Speaker_Embedding(nn.Module):
#     """Disciminative Embedding module:
#         Simple embedding layer with init weight
#     """
#
#     def __init__(self, hparams):
#         super(Speaker_Embedding, self).__init__()
#         self.speaker_embedding = nn.Embedding(hparams.speakers_number, embedding_dim=hparams.speaker_embedding_dims)  ### hparams.speakers_number is number of speakers
#         torch.nn.init.xavier_uniform_(self.speaker_embedding.weight)
#     def forward(self, x):
#         ### x should has [batch, 2] shape because there are 2 speakers
#         embedded = self.speaker_embedding(x)
#         return embedded ### final_outputs.shape[0] = batch size
#
#
# class Language_Embedding(nn.Module):
#     """Language  Embedding module:
#         Simple embedding layer with init weight
#     """
#
#     def __init__(self, hparams):
#         super(Language_Embedding, self).__init__()
#         self.language_embedding = nn.Embedding(hparams.languages_number, embedding_dim=hparams.lang_embedding_dims)  ### hparams.languages_number is number of languages. 32 is embedding dimension
#         torch.nn.init.xavier_uniform_(self.language_embedding.weight)
#     def forward(self, x):
#         ### x should has [batch, 2] shape because there are 2 languages
#         embedded = self.language_embedding(x)
#         return embedded ### final_outputs.shape[0] = batch size
#
#
# class Tacotron2(nn.Module):
#     def __init__(self, hparams):
#         super(Tacotron2, self).__init__()
#         self.mask_padding = hparams.mask_padding
#         self.fp16_run = hparams.fp16_run
#         self.n_mel_channels = hparams.n_mel_channels
#         self.n_frames_per_step = hparams.n_frames_per_step
#         self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
#         std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
#         val = sqrt(3.0) * std  # uniform bounds for std
#         self.embedding.weight.data.uniform_(-val, val)
#         self.speakers_number = hparams.speakers_number
#         self.languages_number = hparams.languages_number
#
#         self.encoder = Encoder(hparams)
#         self.decoder = Decoder(hparams)
#         self.postnet = Postnet(hparams)
#
#     def parse_batch(self, batch):
#         ### format variables' datatype and send them to GPU
#         speaker_ids, text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
#         speaker_ids = to_gpu(speaker_ids).long()
#         text_padded = to_gpu(text_padded).long()
#         input_lengths = to_gpu(input_lengths).long()
#         max_len = torch.max(input_lengths.data).item()
#         mel_padded = to_gpu(mel_padded).float()
#         gate_padded = to_gpu(gate_padded).float()
#         output_lengths = to_gpu(output_lengths).long()
#
#         return ((speaker_ids,text_padded, input_lengths, mel_padded, max_len, output_lengths),
#             (mel_padded, gate_padded))
#
#     def parse_output(self, outputs, output_lengths=None):
#         if self.mask_padding and output_lengths is not None:
#             mask = ~get_mask_from_lengths(output_lengths)
#             mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
#             mask = mask.permute(1, 0, 2)
#
#             outputs[0].data.masked_fill_(mask, 0.0)
#             outputs[1].data.masked_fill_(mask, 0.0)
#             outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
#
#         return outputs
#
#     def forward(self, inputs):
#
#         speaker_ids, text_inputs, text_lengths, mels, max_len, output_lengths = inputs
#         text_lengths, output_lengths = text_lengths.data, output_lengths.data
#         #### todo: with this project. we use only 1 speaker for each language. then language id and speaker id can be used exchangeably.
#         # todo: config this line (and data_utils\TextMelCollate\__call__ function to return language id if there are more than 1 speaker per language.
#         ### until now, speaker_ids has [batch, max_length] shape
#         language_ids = speaker_ids
#
#         #### ENCODING
#         embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
#         encoder_outputs = self.encoder(embedded_inputs, text_lengths)
#         # plt.figure()
#         # plt.imshow(encoder_outputs.float()[0].data.cpu().numpy())
#         # plt.savefig('training_encoder_outputs.png')
#         #### DECODING
#         mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths, speaker_ids=speaker_ids, language_ids=language_ids)
#         mel_outputs_postnet = self.postnet(mel_outputs)
#         mel_outputs_postnet = mel_outputs + mel_outputs_postnet
#         output = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments],output_lengths)
#         return output
#
#     def inference(self, inputs, speaker_ids, language_ids):
#         ### for our project, with each language, we have 1 speaker then speaker and language can be used as the same
#         #### expand language_id to [batch, max_length, language_number] shape
#         language_ids = language_ids.unsqueeze(-1).expand((language_ids.shape[0], inputs.shape[1], self.languages_number))  ### shape: [batch, max_length, language_number]
#         one_hots = torch.zeros(language_ids.size(0), language_ids.size(1), language_ids.size(2)).zero_()  ### zeros's tensor with shape: [batch, max_length, language_number]
#         language_ids = one_hots.scatter_(dim=2, index=language_ids, value=1).to(inputs.device) ### shape: [batch, max_length, language_number]
#         speaker_ids = speaker_ids.to(inputs.device)
#
#
#         #### ENCODING
#         embedded_inputs = self.embedding(inputs).transpose(1, 2)
#         encoder_outputs = self.encoder.inference(embedded_inputs, language_ids=language_ids)
#         ### DECODING
#         mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs, speaker_ids=speaker_ids, language_ids=language_ids)
#         mel_outputs_postnet = self.postnet(mel_outputs)
#         mel_outputs_postnet = mel_outputs + mel_outputs_postnet
#         outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
#         return outputs
