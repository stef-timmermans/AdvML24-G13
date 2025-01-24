import collections
import einops
import tqdm

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding, Layer, Add, MultiHeadAttention, LayerNormalization, Dense, Dropout, StringLookup
from tensorflow.keras import Sequential, Model

from utils.dataset import load_image

class SeqEmbedding(Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()

        self.pos_embedding = Embedding(input_dim=max_length, output_dim=depth)

        self.token_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=depth,
            mask_zero=True
        )

        self.add = Add()

    def call(self, seq):
        seq = self.token_embedding(seq) # (batch, seq, depth)

        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)

        return self.add([seq,x])


class CausalSelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()

        self.mha = MultiHeadAttention(**kwargs)
        # Use Add instead of + so the keras mask propagates through.
        self.add = Add() 
        self.layernorm = LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x,
                        use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)


class CrossAttention(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add() 
        self.layernorm = LayerNormalization()

    def call(self, x, y, **kwargs):
        attn, attention_scores = self.mha(query=x, value=y, return_attention_scores=True)

        self.last_attention_scores = attention_scores

        x = self.add([x, attn])
        return self.layernorm(x)


class FeedForward(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()

        self.seq = Sequential([
            Dense(units=2*units, activation='relu'),
            Dense(units=units),
            Dropout(rate=dropout_rate),
        ])

        self.layernorm = LayerNormalization()

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)


class DecoderLayer(Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()

        self.self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        in_seq, out_seq = inputs

        # Text input
        out_seq = self.self_attention(out_seq)

        out_seq = self.cross_attention(out_seq, in_seq)

        self.last_attention_scores = self.cross_attention.last_attention_scores

        out_seq = self.ff(out_seq)

        return out_seq


class TokenOutput(Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()

        self.dense = Dense(units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens

        self.bias = None

    # "This reduces the initial loss from the entropy of the uniform distribution
    # (log(vocabulary_size)) to the marginal entropy of the distribution (-p*log(p))."
    def adapt(self, ds):
        counts = collections.Counter()

        vocab_dict = {
           name: id for id, name in enumerate(self.tokenizer.get_vocabulary())
        }

        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())

        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr/total
        p[counts_arr==0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p*p).sum()

        print()
        print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        self.bias = log_p
        self.bias[counts_arr==0] = -1e9

    def call(self, x):
        x = self.dense(x)
        # TODO(b/250038731): Fix this.
        # An Add layer doesn't work because of the different shapes.
        # This clears the mask, that's okay because it prevents keras from rescaling
        # the losses.
        return x + self.bias


class Captioner(Model):
    def __init__(
        self, 
        tokenizer, 
        feature_extractor, 
        output_layer, 
        num_layers=1,
        units=256, 
        max_length=50, 
        num_heads=1, 
        dropout_rate=0.1
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.word_to_index = StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary()
        )

        self.index_to_word = StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True
        ) 

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length
        )

        self.decoder_layers = [
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for n in range(num_layers)
        ]

        self.output_layer = output_layer

    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def call(self, inputs):
        image, txt = inputs

        if image.shape[-1] == 3:
            # Apply the feature-extractor, if you get an RGB image.
            image = self.feature_extractor(image)

        # Flatten the feature map
        image = einops.rearrange(image, 'b h w c -> b (h w) c')

        if txt.dtype == tf.string:
            # Apply the tokenizer if you get string inputs.
            txt = self.tokenizer(txt)

        txt = self.seq_embedding(txt)

        # Look at the image
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))

        txt = self.output_layer(txt)

        return txt

    def simple_gen(self, image, temperature=1):
        initial = self.word_to_index([['[START]']]) # (batch, sequence)
        img_features = self.feature_extractor(image[tf.newaxis, ...])

        tokens = initial # (batch, sequence)

        for n in range(50):
            preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
            preds = preds[:,-1, :]  #(batch, vocab)
            if temperature==0:
                next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
            else:
                next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
            tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

            if next[0] == self.word_to_index('[END]'):
                break

        words = self.index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()
    
    # def run_and_show_attention(self, image, temperature=0.0):
    #     result_txt = self.simple_gen(image, temperature)
    #     str_tokens = result_txt.split()
    #     str_tokens.append('[END]')

    #     attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
    #     attention_maps = tf.concat(attention_maps, axis=0)
    #     attention_maps = einops.reduce(
    #         attention_maps,
    #         'batch heads sequence (height width) -> sequence height width',
    #         height=7, width=7,
    #         reduction='mean')

    #     plot_attention_maps(image/255, str_tokens, attention_maps)
    #     t = plt.suptitle(result_txt)
    #     t.set_y(1.05)


class GenerateText(tf.keras.callbacks.Callback):
    def __init__(self, image_shape):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
        self.image = load_image(image_path, image_shape)

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()