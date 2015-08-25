args <- commandArgs(trailingOnly = TRUE)
trainingSet <- read.csv(args[1])
attributes <- read.csv(args[2], header = FALSE)

for (i in 0:9) {
    print(i)
    curTrainingSet <- trainingSet
    binarized.label <- sapply(curTrainingSet$label, function(x) if(x==i){1}else{0})
    curTrainingSet$label <- binarized.label
    pixels <- attributes[i+1,]
    pixels <- pixels[!is.na(pixels)]
    colheaders <- sapply(pixels, function(x) paste("pixel", as.character(x-1), sep=''))
    curTrainingSet <- curTrainingSet[,c("label", colheaders)]
    # model[i] = train an ML algorithm on this training set
}

# for each row in test set, pass it through model[0] through model[9]
# pick highest predicted probability of all the models
