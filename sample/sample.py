import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

# Sample training data
data = "hello"
chars = sorted(list(set(data)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = 4
X_data = []
y_data = []
for i in range(0, len(data) - input_len):
    in_seq = data[i:i + input_len]
    out_seq = data[i + input_len]
    X_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

X = np.reshape(X_data, (len(X_data), input_len, 1))
X = X / float(len(chars))
y = to_categorical(y_data)

# Define LSTM model
model = Sequential([
    LSTM(256, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, batch_size=64, verbose=1)

# Generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = [char_to_num[char] for char in seed_text]
        token_list = np.reshape(token_list, (1, len(token_list), 1))
        token_list = token_list / float(len(chars))
        predicted = model.predict(token_list, verbose=0)
        predicted_class_index = np.argmax(predicted, axis=-1)[0]
        output_char = ""
        for char, index in char_to_num.items():
            if index == predicted_class_index:
                output_char = char
                break
        seed_text += output_char
    return seed_text

print(generate_text("hell", 5, model, input_len))
