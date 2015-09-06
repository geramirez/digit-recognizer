dataSetsPath <- "C:\\Users\\samarths\\Desktop\\newdata\\"

require(randomForest)
rf.models <- list()

parentTestSet <- read.csv(paste(dataSetsPath, "test_FS.csv", sep=''), header = TRUE)
parentTrainingSet <- read.csv(paste(dataSetsPath, "train_FS.csv", sep=''), header=TRUE)
parentTrainingSet <- parentTrainingSet[,-1]
mergedSet <- rbind(parentTrainingSet, parentTestSet)

# PCA on merged set.
pca <- prcomp(mergedSet, scale = TRUE)

# Find 95% threshold of cumulative variance proportion of the principal components.
thresh <- max(which(cumsum(pca$sdev^2 / sum(pca$sdev^2)) <= 0.95))

# for each digit
for (i in 0:9) {
  print(i)
  trainingSet <- read.csv(paste(dataSetsPath, "Train_FS_scaled\\", "trainFS-digit", i, ".csv", sep=''), header=TRUE)

  # Apply PCA to training set.
  train.pca <- predict(pca, trainingSet[,-ncol(trainingSet)])
  
  # Train RF model.
  model <- randomForest(train.pca[,1:thresh], as.factor(trainingSet[,ncol(trainingSet)]))
  
  # Add the RF model to the list of models, one per digit.
  rf.models[[length(rf.models)+1]] <- model
}

testSet <- read.csv(paste(dataSetsPath, "test_FS.csv", sep=''), header = TRUE)
# apply PCA to test set
test.pca <- predict(pca, testSet)

predictions <- matrix(nrow = nrow(testSet), ncol=0)

# for each digit
for (i in 0:9) {
  print(i)
  
  # Predictions from RF for this digit.
  prediction <- predict(rf.models[[i+1]], test.pca[,1:thresh], type = "prob")
  
  # Store predicted probability for this digit.
  predictions <- cbind(predictions, prediction[,"1"])
}

require(MASS)
write.matrix(predictions, paste(dataSetsPath, "probabilities.csv", sep=''), sep=',')
