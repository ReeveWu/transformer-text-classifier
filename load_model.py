from transformers import AutoTokenizer, TFAutoModel
import numpy as np 
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

tf.config.list_physical_devices('GPU')
bert_model = TFAutoModel.from_pretrained('BAAI/bge-base-en-v1.5')

def create_model():
    input_ids = tf.keras.layers.Input(shape=(50, ), name='input_ids', dtype=tf.int32)
    mask = tf.keras.layers.Input(shape=(50, ), name='attention_mask', dtype=tf.int32)
    bert_model.trainable = False
    embeddings = bert_model.bert(input_ids=input_ids, attention_mask=mask)['pooler_output']
    x = tf.keras.layers.Dense(50, activation='relu')(embeddings)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    return model

model = create_model()
model.load_weights("model/v2/v2")

def tokenize(text):
    tokens = tokenizer.encode_plus(text=text, max_length=50, 
                                truncation=True, padding='max_length', 
                                add_special_tokens=True, return_token_type_ids=False, 
                                return_attention_mask=True, return_tensors='tf')
    return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}

def predict(text):
    tokens = tokenize(text)
    results = model.predict(tokens)

    score = np.dot(np.array(results), np.transpose(np.array([1, 2, 3, 4, 5])))/ np.sum(results)
    return score[0]


if __name__ == "__main__":
    result = predict("This is the best video I have ever watched. ")
    print(f"Score: {result}")