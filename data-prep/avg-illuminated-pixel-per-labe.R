#
#  R script to plot pixel density for each label
#

d = train
d2= d[,grep('pixel', colnames(d))]

for (column in 1:ncol(d2)) {
  v = d2[,column]
  b = ifelse(v == 0, 0, 1)
  d2[,column] = b
}
d2$label = d$label


pixel_map <-matrix(data=0.0, nrow=10, ncol=ncol(d2))
for (label in 0:9) {
  df = d2[d2$label == label,]
  row_index = label + 1

  for (pixel in 1:(ncol(d2)-1)) {
    column_index = pixel
    column = df[,column_index]
    s = sum(column)
    value = s / (1.0 * nrow(df))
    pixel_map[row_index, column_index] = value
    print(value)
  }
}

title = 'illuminated pixel percent per label'
for (label in 1:10) {
  plot(pixel_map[label,]+2*(label-1), col=label, xlab='pixel', ylab='Label 0 thru 9', ylim=c(0,20), type='l',  yaxt='n', main=title)
  par(new=TRUE)
}

