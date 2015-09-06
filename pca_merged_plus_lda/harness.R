# Set dataset path: assumes that all data is inside files
dataSetsPath <- "~/Code/UW_Data_Science/DigitRec/digit-recognizer/newdata/"

# Import requirements
require(MASS)

# Initalize lists
models <- list()

# Load test data and predictions array
testData <- read.csv(paste(dataSetsPath, "test_FS.csv", sep=''), header = TRUE)
trainingData <- read.csv(paste(dataSetsPath, "train_FS.csv", sep=''), header=TRUE)
predictions <- matrix(nrow = nrow(testData), ncol=0)
validationData <- trainingData[35000:nrow(trainingData),]

# set classifier
classifier <- lda
trainModel <- function(data, label) {
  return(classifier(data, label, probability=TRUE))
}

# Initalize functions
loadDigitData <- function(segment, number){
  # Loads individual digit data
  path <- paste(dataSetsPath, segment, number, ".csv", sep='')
  return(read.csv(path, header=TRUE))
}

# Calculate PCA
mergedSet <- rbind(testData, trainingData[,2:ncol(trainingData)])
# PCA on merged set.
pca <- prcomp(mergedSet, scale = TRUE)
# Find 95% threshold of cumulative variance proportion of the principal components.
thresh <- max(which(cumsum(pca$sdev^2 / sum(pca$sdev^2)) <= 0.95))


# for each digit
for (digit in 0:9) {
  # Load data
  trainingSet <- loadDigitData("Train_FS_scaled/trainFS-digit", digit)[1:10000, ]
  # Apply PCA
  train.pca <- predict(pca, trainingSet[,-ncol(trainingSet)])
  # Train SVM model.
  model <- trainModel(train.pca[,1:thresh],
                      as.factor(trainingSet[,ncol(trainingSet)]))
  # Add the SVM model to the list of models, one per digit.
  models[[digit+1]] <- model
  print(digit)
}

# Validate model
validationPredictions <- matrix(nrow = nrow(validationData), ncol=0)
validate.pca <- predict(pca, validationData[,2:ncol(validationData)])
for (digit in 0:9) {
  # Make digit prediction
  prediction <- predict(models[[digit+1]], validate.pca[,1:thresh], probability = TRUE)
  # Store predicted probability for this digit.
  validationPredictions <- cbind(validationPredictions, prediction$posterior[,2])
  print(digit)
}
validationPredictions <- data.frame(validationPredictions)
validationPredictions['prediction'] <- max.col(validationPredictions[1:10])- 1
validationPredictions['label'] <- validationData$label
validationPredictions['ImageId'] <- 1:nrow(validationPredictions)
table(validationPredictions['label'] == validationPredictions['prediction']) / nrow(validationPredictions)


# Make predictions on test data
predictions <- matrix(nrow = nrow(testData), ncol=0)
test.pca <- predict(pca, testData[,-ncol(trainingSet)])
for (digit in 0:9) {
  # Make digit prediction
  prediction <- predict(models[[digit+1]], test.pca[,1:thresh], probability = TRUE)
  # Store predicted probability for this digit.
  predictions <- cbind(predictions, prediction$posterior[,2])
  print(digit)
}

predictions <- data.frame(predictions)
predictions['Label'] <- max.col(predictions[1:10])- 1
predictions['ImageId'] <- 1:nrow(predictions)
write.csv(predictions[11:12], paste(dataSetsPath, "probabilities_results_new.csv", sep=''))

