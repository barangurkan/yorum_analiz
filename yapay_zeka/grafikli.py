import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import re
import string
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback


# Duygu tahmin fonksiyonu
def predict_sentiment(model, tokenizer, max_tokens, text, threshold=0.5):
    text_tokens = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_tokens, maxlen=max_tokens)

    prediction = model.predict(text_pad)[0]
    positive_prob = prediction[2] * 100
    neutral_prob = prediction[1] * 100
    negative_prob = prediction[0] * 100

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


class CustomCallback(Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

        with open(self.log_file, "w") as f:
            f.write("Egitim Logları Basladi:\n")
            f.write("Epoch\tLoss\tAccuracy\tVal_Loss\tVal_Accuracy\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0)
        accuracy = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_accuracy = logs.get('val_accuracy', 0)

        print(
            f"Epoch {epoch + 1}: Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Dosyaya yaz
        with open(self.log_file, "a") as f:
            f.write(f"{epoch + 1}\t{loss:.4f}\t{accuracy:.4f}\t{val_loss:.4f}\t{val_accuracy:.4f}\n")


# Ana fonksiyon
def main(train_model=True):
    if train_model:
        df1 = pd.read_csv("C:/Users/dogab/OneDrive/Desktop/YapayZeka___PythonProjesi/duzenlenmis_veriler.csv",
                          delimiter=';')
        df2 = pd.read_csv("C:/Users/dogab/OneDrive/Desktop/YapayZeka___PythonProjesi/veriler1.csv", delimiter=';')

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

        model.fit(
            X_train_pad,
            np.array(y_train),
            epochs=5,
            batch_size=64,
            validation_data=(X_test_pad, np.array(y_test)),
            callbacks=[CustomCallback("training_logs.txt")]  # Log dosyasını belirtin
        )

        model.save("model.h5")
        with open("tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        model = load_model("model.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)

        max_tokens = 100

        print("--------------------------------------------------------")
        user_input = input("Lütfen bir metin girin: ")

        # Tahmin ve eşik değeri ile sonucu hesapla
        predicted_sentiment, probability = predict_sentiment(model, tokenizer, max_tokens, user_input, threshold=0.45)
        print(f"Girilen metnin duygusu: {predicted_sentiment} -> Olasılık: % {probability:.2f}\n")


if __name__ == "__main__":
    main(train_model=True)
