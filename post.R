library(h5)
library(ggplot2)

vac.scatterplot <- function(filename) {

  file <- h5file(filename,"r")
  dset <- file["means"]
  dat <- data.frame(β = c(dset[1,]), mean = c(dset[2,]), t = c(dset[3,]))
  dat
}
