import os
import numpy as np
import h5py
import pandas as pd
import string
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_sentiment(model, tokenizer, max_tokens, text, threshold=0.5):
    text_tokens = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)

    prediction = model.predict(text_pad)[0]
    positive_prob = prediction[2] * 100  # Olumlu olasılık
    neutral_prob = prediction[1] * 100  # Nötr olasılık
    negative_prob = prediction[0] * 100  # Olumsuz olasılık

    if negative_prob >= 51:
        sentiment = 'Olumsuz 😞'
        probability = negative_prob
    elif positive_prob >= 51:
        sentiment = 'Olumlu 😀👍🏻'
        probability = positive_prob
    else:
        sentiment = 'Nötr 😐'
        probability = neutral_prob

    return sentiment, probability


def remove_punctuation(text):
    no_punc = [char for char in text if char not in string.punctuation]
    return "".join(no_punc)


def remove_numeric(corpus):
    return "".join(words for words in corpus if not words.isdigit())


def main(train_model=True):
    if train_model:
        # Eğitim için gerekli veriler
        df1 = pd.read_csv("duzenlenmis_veriler.csv", delimiter=';')
        df2 = pd.read_csv("veriler1.csv", delimiter=';')

        df = pd.concat([df1, df2], ignore_index=True)

        df["comment"] = df["Metin"].apply(lambda x: str(x).lower())
        df["comment"] = df["comment"].apply(remove_punctuation)
        df["comment"] = df["comment"].apply(lambda x: x.replace("\r", " ").replace("\n", " "))
        df["comment"] = df["comment"].apply(remove_numeric)

        df["point"] = df["Durum"].replace({1: 2, 2: 0, 4: 1, 5: 1})

        data = df["comment"].values.tolist()
        target = df["point"].values.tolist()

        cutoff = int(len(data) * 0.80)
        X_train, X_test = data[:cutoff], data[cutoff:]
        y_train, y_test = target[:cutoff], target[cutoff:]

        num_words = 10000
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(data)

        X_train_tokens = tokenizer.texts_to_sequences(X_train)
        X_test_tokens = tokenizer.texts_to_sequences(X_test)

        num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
        num_tokens = np.array(num_tokens)
        max_tokens = int(np.mean(num_tokens) + (2 * np.std(num_tokens)))

        X_train_pad = pad_sequences(X_train_tokens, maxlen=max_tokens)
        X_test_pad = pad_sequences(X_test_tokens, maxlen=max_tokens)

        embedding_size = 50
        model = Sequential()
        model.add(
            Embedding(input_dim=num_words, output_dim=embedding_size, input_length=max_tokens, name="embedding_layer"))
        model.add(GRU(units=16, return_sequences=True))
        model.add(GRU(units=8, return_sequences=True))
        model.add(GRU(units=4))
        model.add(Dense(3, activation="softmax"))

        optimizer = Adam(learning_rate=1e-3)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        model.fit(X_train_pad, np.array(y_train), epochs=2, batch_size=64,
                  validation_data=(X_test_pad, np.array(y_test)))

        model.save("model.h5")
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("max_tokens.pickle", "wb") as handle:
            pickle.dump(max_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Tahmin yapma işlemi
        model = load_model("model.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        with open("max_tokens.pickle", "rb") as handle:
            max_tokens = pickle.load(handle)

        print("--------------------------------------------------------")
        user_input = input("Lütfen bir metin girin: ")

        predicted_sentiment, probability = predict_sentiment(model, tokenizer, max_tokens, user_input, threshold=0.45)
        print(f"Girilen metnin duygusu: {predicted_sentiment} -> Olasılık: % {probability:.2f}\n")

if __name__ == "__main__":
    main(train_model=False)  # Eğitim yapmadan tahmin yapmak için False olarak değiştir!
