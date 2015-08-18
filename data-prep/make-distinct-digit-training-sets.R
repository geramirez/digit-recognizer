#
#   Generate distinct trainings sets for use with learnings
#   specialized to single digits
#
#   create distinct training sets for each digit
#   training sets have the label attribute replcaced with a
#   binarized 'digit#' attribute.  There is one training set
#   for each digit.
#
setwd('~/Desktop/DataScience/UW-DataScience-450/kaggle_comp/')
train <- read.csv("~/Desktop/DataScience/UW-DataScience-450/kaggle_comp/train.csv", stringsAsFactors=FALSE)

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
write.csv(train, file = "train-digit0.csv", row.names = FALSE)

train$digit0 = NULL
train$digit1 = digit1
write.csv(train, file = "train-digit1.csv", row.names = FALSE)
rm(digit0)

train$digit1 = NULL
train$digit2 = digit2
write.csv(train, file = "train-digit2.csv", row.names = FALSE)
rm(digit1)

train$digit2 = NULL
train$digit3 = digit3
write.csv(train, file = "train-digit3.csv", row.names = FALSE)
rm(digit2)

train$digit3 = NULL
train$digit4 = digit4
write.csv(train, file = "train-digit4.csv", row.names = FALSE)
rm(digit3)

train$digit4 = NULL
train$digit5 = digit5
write.csv(train, file = "train-digit5.csv", row.names = FALSE)
rm(digit4)


train$digit5 = NULL
train$digit6 = digit6
write.csv(train, file = "train-digit6.csv", row.names = FALSE)
rm(digit5)


train$digit6 = NULL
train$digit7 = digit7
write.csv(train, file = "train-digit7.csv", row.names = FALSE)
rm(digit6)

train$digit7 = NULL
train$digit8 = digit8
write.csv(train, file = "train-digit8.csv", row.names = FALSE)
rm(digit7)


train$digit8 = NULL
train$digit9 = digit9
write.csv(train, file = "train-digit9.csv", row.names = FALSE)
rm(digit8)
rm(digit9)



