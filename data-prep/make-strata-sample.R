#
# Data Prep Script - generate sample
#

sampleDataSet <- function(dataSet, numSamples = 100 )
{
  randoms <- runif(nrow(dataSet))
  fractionOfTest = numSamples / nrow(dataSet)
  cutoff <- quantile(randoms, fractionOfTest)
  testFlag <- randoms <= cutoff
  sampleSet <- dataSet[testFlag, ]
  return(sampleSet)
}


train.0 = train[train$label == 0,]
train.1 = train[train$label == 1,]
train.2 = train[train$label == 2,]
train.3 = train[train$label == 3,]
train.4 = train[train$label == 4,]
train.5 = train[train$label == 5,]
train.6 = train[train$label == 6,]
train.7 = train[train$label == 7,]
train.8 = train[train$label == 8,]
train.9 = train[train$label == 9,]

N = nrow(train)
p = c(
      p.0 = nrow(train.0)/N,
      p.1 = nrow(train.1)/N,
      p.2 = nrow(train.2)/N,
      p.3 = nrow(train.3)/N,
      p.4 = nrow(train.4)/N,
      p.5 = nrow(train.5)/N,
      p.6 = nrow(train.6)/N,
      p.7 = nrow(train.7)/N,
      p.8 = nrow(train.8)/N,
      p.9 = nrow(train.9)/N
)


#
#  Set overall sample size N (truncation makes overall sample a little smaller)
#
N = 2000

#
#  generate array of sample sizes for each stata
#
n = N * p
n = as.integer(n)

s.0 = sampleDataSet(train.0, n[1])
s.1 = sampleDataSet(train.1, n[2])
s.2 = sampleDataSet(train.2, n[3])
s.3 = sampleDataSet(train.3, n[4])
s.4 = sampleDataSet(train.4, n[5])
s.5 = sampleDataSet(train.5, n[6])
s.6 = sampleDataSet(train.6, n[7])
s.7 = sampleDataSet(train.7, n[8])
s.8 = sampleDataSet(train.8, n[9])
s.9 = sampleDataSet(train.9, n[10])

#
#  join stata into single data frame
#
train.sample = rbind(s.0, s.1)
train.sample = rbind(train.sample, s.2)
train.sample = rbind(train.sample, s.3)
train.sample = rbind(train.sample, s.4)
train.sample = rbind(train.sample, s.5)
train.sample = rbind(train.sample, s.6)
train.sample = rbind(train.sample, s.7)
train.sample = rbind(train.sample, s.8)
train.sample = rbind(train.sample, s.9)

#
#  Save for later...
#
write.csv(train.sample, file = "train-sample.csv", col.names = TRUE, row.names = FALSE)
