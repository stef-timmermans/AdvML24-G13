import collections
import einops
import tqdm
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding, Layer, Add, MultiHeadAttention, LayerNormalization, Dense, Dropout, StringLookup, TextVectorization
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import Callback

from utils.dataset import load_image 

# Model Architecture, source: https://www.tensorflow.org/text/tutorials/image_captioning#a_transformer_decoder_model
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

        x = tf.range(tf.shape(seq)[1])
        x = x[tf.newaxis, :]
        x = self.pos_embedding(x)

        return self.add([seq,x])
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "depth": self.depth
        }


class CausalSelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()

        self.mha = MultiHeadAttention(**kwargs)
        self.add = Add() 
        self.layernorm = LayerNormalization()

    def call(self, x):
        attn = self.mha(query=x, value=x,
                        use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)
    
    def get_config(self):
        return self.mha.get_config()


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
    
    def get_config(self):
        return self.mha.get_config()


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
    
    def get_config(self):
        return {
            "units": self.units,
            "dropout_rate": self.dropout_rate
        }


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
    
    def get_config(self):
        return {
            "units": self.units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        }


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
        return x + self.bias
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tokenizer": {
                "max_tokens": self.tokenizer.vocabulary_size(),
                "output_sequence_length": 50,
                "standardize": standardize,
                "ragged": True,
                "vocabulary": self.tokenizer.get_vocabulary()
            },
            "banned_tokens": self.banned_tokens
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        tokenizer_config = config.pop("tokenizer")

        tokenizer = TextVectorization(
            max_tokens=tokenizer_config["max_tokens"],
            output_sequence_length=tokenizer_config["output_sequence_length"],
            standardize=standardize,
            ragged=tokenizer_config["ragged"]
        )
        tokenizer.set_vocabulary(tokenizer_config["vocabulary"])

        return cls(tokenizer=tokenizer, **config)


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

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.units = units
        self.max_length = max_length
        self.dropout_rate = dropout_rate

        self.output_layer = output_layer
    
    def call(self, inputs):
        image, txt = inputs

        # Check if image
        if image.shape[-1] == 3:
            # Use CNN on image
            image = self.feature_extractor(image)

        # Flatten the feature map
        image = einops.rearrange(image, 'b h w c -> b (h w) c')

        # Check if string
        if txt.dtype == tf.string:
            # Apply the tokenizer on string
            txt = self.tokenizer(txt)

        txt = self.seq_embedding(txt)

        # Look at the image
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))

        txt = self.output_layer(txt)

        return txt
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "tokenizer": {
                "max_tokens": self.tokenizer.vocabulary_size(),
                "output_sequence_length": 50,
                "standardize": standardize,  # Save function name
                "ragged": True,
                "vocabulary": self.tokenizer.get_vocabulary()
            },
            "feature_extractor": tf.keras.utils.serialize_keras_object(self.feature_extractor),
            "output_layer": tf.keras.utils.serialize_keras_object(self.output_layer),
            "word_to_index": tf.keras.utils.serialize_keras_object(self.word_to_index),
            "index_to_word": tf.keras.utils.serialize_keras_object(self.index_to_word),
            "num_layers": self.num_layers,
            "units": self.units,
            "max_length": self.max_length,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        tokenizer_config = config.pop("tokenizer")
        # standardize_function = globals().get("standardize")
        
        tokenizer = TextVectorization(
            max_tokens=tokenizer_config["max_tokens"],
            output_sequence_length=tokenizer_config["output_sequence_length"],
            standardize=standardize,
            ragged=tokenizer_config["ragged"]
        )
        tokenizer.set_vocabulary(tokenizer_config["vocabulary"])

        feature_extractor = tf.keras.utils.deserialize_keras_object(config.pop("feature_extractor"))
        output_layer = tf.keras.utils.deserialize_keras_object(config.pop("output_layer"))
        
        return cls(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            output_layer=output_layer,
            num_layers=config["num_layers"],
            units=config["units"],
            max_length=config["max_length"],
            num_heads=config["num_heads"],
            dropout_rate=config["dropout_rate"]
        )

    def simple_gen(self, image, temperature=1):
        initial = self.word_to_index([['[START]']])
        img_features = self.feature_extractor(image[tf.newaxis, ...])

        tokens = initial

        for n in range(50):
            preds = self((img_features, tokens)).numpy()
            preds = preds[:,-1, :]
            if temperature==0:
                next = tf.argmax(preds, axis=-1)[:, tf.newaxis]
            else:
                next = tf.random.categorical(preds/temperature, num_samples=1)
            tokens = tf.concat([tokens, next], axis=1)

            if next[0] == self.word_to_index('[END]'):
                break

        words = self.index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()
    
    def run_and_show_attention(self, image, temperature=0.0):
        result_txt = self.simple_gen(image, temperature)
        str_tokens = result_txt.split()
        str_tokens.append('[END]')

        attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
        attention_maps = tf.concat(attention_maps, axis=0)
        attention_maps = einops.reduce(
        attention_maps,
        'batch heads sequence (height width) -> sequence height width',
        height=7, width=7,
        reduction='mean')

        plot_attention_maps(image/255, str_tokens, attention_maps)
        t = plt.suptitle(result_txt)
        t.set_y(1.05)


class GenerateText(Callback):
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


# Graph function
def plot_attention_maps(image, str_tokens, attention_map):
    fig = plt.figure(figsize=(16, 9))

    len_result = len(str_tokens)

    titles = []
    for i in range(len_result):
      map = attention_map[i]
      grid_size = max(int(np.ceil(len_result/2)), 2)
      ax = fig.add_subplot(3, grid_size, i+1)
      titles.append(ax.set_title(str_tokens[i]))
      img = ax.imshow(image)
      ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(),
                clim=[0.0, np.max(map)])

    plt.tight_layout()


# Custom standardize function for TextVectorization layer
@tf.keras.utils.register_keras_serializable()
def standardize(s):
    s = tf.strings.lower(s) # Lowercase
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '') # Remove puncuation
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ') # Add [START] and [END] tokens to the text
    return s