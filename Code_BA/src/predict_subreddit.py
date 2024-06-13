import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert categorical labels to numerical (e.g., set_A = 0, set_B = 1)
    df['set'] = df['set'].apply(lambda x: 0 if x == 'set_A' else 1)

    # delete word duplicates
    df = df.drop_duplicates(subset='word', keep='first')
    
    # Pivot the data to have dimensions as columns
    df_pivot = df.pivot(index='word', columns='dimension', values='distance').reset_index()
    df_labels = df[['word', 'set']].drop_duplicates()

    # Merge to get a single DataFrame
    df_merged = pd.merge(df_pivot, df_labels, on='word')

    # Features and labels
    X = df_merged.drop(columns=['word', 'set'])
    y = df_merged['set']

    return X, y

def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def main():
    name = 'connotations_try'
    # Load and preprocess data
    X, y = load_and_preprocess_data(f'data/data_helper/connotations/{name}.csv')

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to categorical (one-hot encoding)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Build the ANN
    model = build_ann(input_dim=X_train.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
