import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold

# Load data (modified column names kept as per your example)
def loadData(csvPath):
    dataFrame = pd.read_csv(csvPath)
    inputFeatures = dataFrame[['BA.x', 'OBP.x', 'SLG.x']].values
    outputTargets = dataFrame[['BA.y', 'OBP.y', 'SLG.y']].values
    return inputFeatures, outputTargets

def createDNNModel(inputShape):
    featureNormalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(inputShape,)),
        featureNormalizer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # 20% dropout after first hidden layer
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # 20% dropout after second hidden layer
        tf.keras.layers.Dense(3, activation='linear')  # Output layer (no dropout here)
    ])
    return model, featureNormalizer

def train(csvPath, epochs, numFolds, alpha):
    # Load and split data
    allFeatures, allTargets = loadData(csvPath)
    xTrain, xTest, yTrain, yTest = train_test_split(
        allFeatures, allTargets, test_size=0.2, random_state=42
    )

    # Initialize KFold
    kFold = KFold(n_splits=numFolds, shuffle=True, random_state=42)
    foldNumber = 0
    foldLosses = []
    foldMaes = []

    # K-fold cross validation
    for trainIndices, valIndices in kFold.split(xTrain):
        foldNumber += 1
        print(f"\nTraining fold {foldNumber}/{numFolds}")
        
        # Split fold data
        xTrainFold = xTrain[trainIndices]
        yTrainFold = yTrain[trainIndices]
        xValFold = xTrain[valIndices]
        yValFold = yTrain[valIndices]

        # Create and adapt model for this fold
        currentModel, normalizer = createDNNModel(xTrain.shape[1])
        normalizer.adapt(xTrainFold)
        
        currentModel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
            loss='mse',
            metrics=['mae']
        )

        # Train with validation data
        currentModel.fit(
            xTrainFold, yTrainFold,
            validation_data=(xValFold, yValFold),
            epochs=epochs,
            verbose=1
        )

        # Evaluate fold performance
        valLoss, valMae = currentModel.evaluate(xValFold, yValFold, verbose=0)
        foldLosses.append(valLoss)
        foldMaes.append(valMae)
        print(f"Fold {foldNumber} validation loss: {valLoss:.4f}, MAE: {valMae:.4f}")

    # Print cross-validation results
    print("\nCross-validation results:")
    print(f"Average loss: {np.mean(foldLosses):.4f} (±{np.std(foldLosses):.4f})")
    print(f"Average MAE: {np.mean(foldMaes):.4f} (±{np.std(foldMaes):.4f})")

    # Train final model on entire dataset
    print("\nTraining final model on entire training set...")
    finalModel, finalNormalizer = createDNNModel(xTrain.shape[1])
    finalNormalizer.adapt(xTrain)
    finalModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
        loss='mse',
        metrics=['mae']
    )
    finalModel.fit(xTrain, yTrain, epochs=epochs, verbose=1)

    # Evaluate on test set
    testLoss, testMae = finalModel.evaluate(xTest, yTest, verbose=0)
    print(f"\nFinal test loss: {testLoss:.4f}")
    print(f"Final test MAE: {testMae:.4f}")
    print(f"numEpochs: {epochs}")
    print(f"Learning rate: {alpha}")

    # Save model
    # finalModel.save("tfLinearWithPreprocessing")
    # print("\nModel saved as 'tfLinearWithPreprocessing'")
    return finalModel, testLoss, testMae

def predictWithModel(modelPath, newData):
    loadedModel = tf.keras.models.load_model(modelPath)
    return loadedModel.predict(newData)

if __name__ == "__main__":
    epochs = 50
    numFolds = 5
    alpha = 0.1

    alphas = [0.001,0.01,0.1,0.2,0.5]

    #   Implement nested loop that goes through and tests each combination
    #   of hyperparameters to find optimal
    results = []

    for e in range(10, 110, 10):
        for a in alphas:
            finalModel, testLoss, testMae = train('Data/commonCleaned.csv', e, numFolds, a)
            # Append the results to the list
            results.append({'numEpochs': e, 'learningRate': a, 'testLoss': testLoss, 'testMae': testMae})

    # Convert the list to a DataFrame
    resultsDF = pd.DataFrame(results)

    # Save the DataFrame to a CSV file with an index
    resultsDF.to_csv('Training Evaluation/training_results_DNN.csv', index=False)


    # trainedModel = train('Data/commonCleaned.csv', epochs, numFolds, alpha)
    
    # Example prediction
    # sampleInput = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)
    # prediction = predictWithModel("tfLinearWithPreprocessing", sampleInput)
    # print(f"\nSample prediction: {prediction}")