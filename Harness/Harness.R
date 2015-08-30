args <- commandArgs(trailingOnly = TRUE)
trainingSet <- read.csv(args[1])
attributes <- read.csv(args[2], header = FALSE)

require(e1071)
models <- list()

# for each digit
for (i in 0:9) {
    print(i)
    
    # create training set with columns identified by the attributes input
    # and with binarized labels.
    curTrainingSet <- trainingSet
    binarized.label <- sapply(curTrainingSet$label, function(x) if(x==i){1}else{0})
    curTrainingSet$label <- binarized.label
    pixels <- attributes[i+1,]
    pixels <- pixels[!is.na(pixels)]
    colheaders <- sapply(pixels, function(x) paste("pixel", as.character(x-1), sep=''))
    curTrainingSet <- curTrainingSet[,c("label", colheaders)]
    curTrainingSet$label <- as.factor(curTrainingSet$label)
    
    # model[i] = train an ML algorithm on this training set
    model <- svm(label~., curTrainingSet, probability = TRUE)
    models[[length(models)+1]] <- model
}

testSet <- read.csv(args[3])
# for each row in test set, pass it through model[0] through model[9]
# pick highest predicted probability of all the models
predictions <- matrix(nrow = nrow(testSet), ncol=0)

for (i in 0:9) {
  
    # reduce test set to same columns as training set, identified by the 
    # attributes input.
    curTestSet <- testSet
    pixels <- attributes[i+1,]
    pixels <- pixels[!is.na(pixels)]
    colheaders <- sapply(pixels, function(x) paste("pixel", as.character(x-1), sep=''))
    curTestSet <- curTestSet[,c(colheaders)]
    
    print(summary(models[[i+1]]))
    
    # predict test set based on training set.
    prediction <- predict(models[[i+1]], curTestSet, probability = TRUE)
    predictions <- cbind(predictions, attr(prediction, "probabilities")[,"1"])
}

require(MASS)
write.matrix(predictions, args[4], sep=',')
