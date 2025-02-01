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

def createDNNModelSmall(inputShape):
    featureNormalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(inputShape,)),
        featureNormalizer,
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer remains linear
    ])
    return model, featureNormalizer

def createDNNModelMedium(inputShape):
    featureNormalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(inputShape,)),
        featureNormalizer,
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer remains linear
    ])
    return model, featureNormalizer

def createDNNModelLarge(inputShape):
    featureNormalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(inputShape,)),
        featureNormalizer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer remains linear
    ])
    return model, featureNormalizer

def createDNNModelDeep(inputShape):
    featureNormalizer = tf.keras.layers.Normalization(axis=-1)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(inputShape,)),
        featureNormalizer,
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # Output layer remains linear
    ])
    return model, featureNormalizer

def train(csvPath, epochs, numFolds, alpha, model_func):
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
        currentModel, normalizer = model_func(xTrain.shape[1])
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
    finalModel, finalNormalizer = model_func(xTrain.shape[1])
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
    numFolds = 5
    alpha = 0.01

    # Prepare model version dictionary
    model_creators = {
        "small": createDNNModelSmall,
        "medium": createDNNModelMedium,
        "large": createDNNModelLarge,
        "deep": createDNNModelDeep
    }

    results = []
    # Try epochs from 10 to 100 in steps of 10
    for e in range(1, 50):
        for model_name, model_func in model_creators.items():
            print(f"----- Training {model_name} -----")
            finalModel, testLoss, testMae = train('Data/commonCleaned.csv', e, numFolds, alpha, model_func)
            results.append({
                'modelUsed': model_name,
                'numEpochs': e,
                'testLoss': testLoss,
                'testMae': testMae
            })
            print(f"----- Results For {model_name} -----")
            print(f"Test Loss: {testLoss}")
            print(f"Test MAE: {testMae}")



    resultsDF = pd.DataFrame(results)
    resultsDF.to_csv('Training Evaluation/DNN_Width_Micro.csv', index=False)