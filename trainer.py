import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH="Data/commonCleaned.csv"
EPOCHS=20

# Load and prepare data
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    
    # Modify these column names according to your CSV structure
    input_columns = ['BA.x', 'OBP.x', 'SLG.x']  # Replace with your input column names
    output_columns = ['BA.y', 'OBP.y', 'SLG.y']  # Replace with your output column names
    
    X = data[input_columns].values
    y = data[output_columns].values
    return X, y

# Build and compile the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)  # Three output nodes
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    return model

# Main training function
def train_model(csv_path, epochs=100, batch_size=32):
    # Load data
    X, y = load_data(csv_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    # Create and train model
    model = create_model()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save model
    model.save("multi_output_model.keras")
    print("Model saved as multi_output_model.keras")
    
    return model, history

# Example usage
if __name__ == "__main__":
    # Replace 'your_data.csv' with your CSV file path
    trained_model, training_history = train_model(
        csv_path=DATA_PATH,
        epochs=EPOCHS,
        batch_size=32
    )