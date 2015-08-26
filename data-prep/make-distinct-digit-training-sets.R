#
#   Generate distinct trainings sets for use with learnings
#   specialized to single digits
#
#   create distinct training sets for each digit
#   training sets have the label attribute replcaced with a
#   binarized 'digit#' attribute.  There is one training set
#   for each digit.
#
setwd('~/Desktop/DataScience/UW-DataScience-450/capstone/datasets/')

#
# Mq <-matrix(data=NA, nrow=ncol(train)-1, ncol=11)
# for (pixel in 2:ncol(train)) {
#   q = quantile(train[,pixel], probs=seq(0,1,1/10))
#   for (percentile in 1:length(q)) {
#     Mq[pixel-1, percentile] = q[percentile]
#   }
# }


sampleDataSet <- function(dataSet, numSamples = 10000)
{
  randoms <- runif(nrow(dataSet))
  fractionOfTest = numSamples / nrow(dataSet)
  cutoff <- quantile(randoms, fractionOfTest)
  testFlag <- randoms <= cutoff
  sampleSet <- dataSet[testFlag, ]
  return(sampleSet)
}

train <- read.csv("~/Desktop/DataScience/UW-DataScience-450/capstone/data-prep/train_RS.csv", stringsAsFactors=FALSE)

digit0 = as.integer(train$label == 0)
digit1 = as.integer(train$label == 1)
digit2 = as.integer(train$label == 2)
digit3 = as.integer(train$label == 3)
digit4 = as.integer(train$label == 4)
digit5 = as.integer(train$label == 5)
digit6 = as.integer(train$label == 6)
digit7 = as.integer(train$label == 7)
digit8 = as.integer(train$label == 8)
digit9 = as.integer(train$label == 9)

train$label=NULL
train$digit0 = digit0
write.csv(train, file = "trainRS-digit0.csv", row.names = FALSE)

train$digit0 = NULL
train$digit1 = digit1
write.csv(train, file = "trainRS-digit1.csv", row.names = FALSE)
rm(digit0)

train$digit1 = NULL
train$digit2 = digit2
write.csv(train, file = "trainRS-digit2.csv", row.names = FALSE)
rm(digit1)

train$digit2 = NULL
train$digit3 = digit3
write.csv(train, file = "trainRS-digit3.csv", row.names = FALSE)
rm(digit2)

train$digit3 = NULL
train$digit4 = digit4
write.csv(train, file = "trainRS-digit4.csv", row.names = FALSE)
rm(digit3)

train$digit4 = NULL
train$digit5 = digit5
write.csv(train, file = "trainRS-digit5.csv", row.names = FALSE)
rm(digit4)


train$digit5 = NULL
train$digit6 = digit6
write.csv(train, file = "trainRS-digit6.csv", row.names = FALSE)
rm(digit5)


train$digit6 = NULL
train$digit7 = digit7
write.csv(train, file = "trainRS-digit7.csv", row.names = FALSE)
rm(digit6)

train$digit7 = NULL
train$digit8 = digit8
write.csv(train, file = "trainRS-digit8.csv", row.names = FALSE)
rm(digit7)


train$digit8 = NULL
train$digit9 = digit9
write.csv(train, file = "trainRS-digit9.csv", row.names = FALSE)
rm(digit8)
rm(digit9)


#
#===========================================================================
#
train <- read.csv("~/Desktop/DataScience/UW-DataScience-450/capstone/data-prep/train_FS.csv", stringsAsFactors=FALSE)

digit0 = as.integer(train$label == 0)
digit1 = as.integer(train$label == 1)
digit2 = as.integer(train$label == 2)
digit3 = as.integer(train$label == 3)
digit4 = as.integer(train$label == 4)
digit5 = as.integer(train$label == 5)
digit6 = as.integer(train$label == 6)
digit7 = as.integer(train$label == 7)
digit8 = as.integer(train$label == 8)
digit9 = as.integer(train$label == 9)

train$label=NULL
train$digit0 = digit0
write.csv(train, file = "trainFS-digit0.csv", row.names = FALSE)

train$digit0 = NULL
train$digit1 = digit1
write.csv(train, file = "trainFS-digit1.csv", row.names = FALSE)
rm(digit0)

train$digit1 = NULL
train$digit2 = digit2
write.csv(train, file = "trainFS-digit2.csv", row.names = FALSE)
rm(digit1)

train$digit2 = NULL
train$digit3 = digit3
write.csv(train, file = "trainFS-digit3.csv", row.names = FALSE)
rm(digit2)

train$digit3 = NULL
train$digit4 = digit4
write.csv(train, file = "trainFS-digit4.csv", row.names = FALSE)
rm(digit3)

train$digit4 = NULL
train$digit5 = digit5
write.csv(train, file = "trainFS-digit5.csv", row.names = FALSE)
rm(digit4)


train$digit5 = NULL
train$digit6 = digit6
write.csv(train, file = "trainFS-digit6.csv", row.names = FALSE)
rm(digit5)


train$digit6 = NULL
train$digit7 = digit7
write.csv(train, file = "trainFS-digit7.csv", row.names = FALSE)
rm(digit6)

train$digit7 = NULL
train$digit8 = digit8
write.csv(train, file = "trainFS-digit8.csv", row.names = FALSE)
rm(digit7)


train$digit8 = NULL
train$digit9 = digit9
write.csv(train, file = "trainFS-digit9.csv", row.names = FALSE)
rm(digit8)
rm(digit9)

