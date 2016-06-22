library(h5)
library(ggplot2)
library(stringr)

## Utility function to read a HDF5 file and create a data frame with it.
read.h5 <- function(filename) {

  file <- h5file(filename,"r")
  dset <- file["means"]
  dat <- data.frame(β = c(dset[,1]), mean = c(dset[,2]), t = c(dset[,3]))
  dat
}

## Plots each field mean value for each temperature as a dot.
mean.scatterplot <- function(filename) {

  dat <- read.h5(filename)

  dev.new(width = 5.5, height = 4.5)
  par(mar = c(4,5,1.5,1.5) + 0.1)
  breaks = c(1/4, 5/16, 3/8, 1/2 , 5/8, 3/4, 1.0, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 6, 8, 10)
  ggplot(dat, aes(x = β, y = mean)) + geom_point(alpha = 0.2) +
         scale_x_log10(expression(beta), breaks = breaks, minor_breaks = NULL) +
         theme(axis.text.x = element_text(angle = -45, vjust = 0.5)) +
         ylab(expression(symbol("\341")~Phi[H]~symbol("\361")[list(tau,x)]))
}

## Log-log plot of the computation time per configuration as a function of the lattice size Ni.
scaling <- function(filename) {

  dat <- read.table(filename, header = TRUE)
  dev.new(width = 4.5, height = 3.5)
  par(mar = c(4,5,1.5,1.5) + 0.1)
  breaks = 16*(1:5)
  ggplot(data = dat, aes(x = Ni, y = t)) + geom_point() +
         geom_errorbar(aes(ymin = t-sd, ymax = t+sd), width=0.01) +
         scale_x_log10(expression(N[i]), breaks = breaks) + scale_y_log10("t [s]")
}

analyze.out <- function(filename) {

  raw_data <- readLines(filename)
  extracted <- str_match(raw_data, "^\\[(\\d):.*\\(ζ = (-?0\\.\\d{3})\\).*\\(ΔS = (-?\\d+\\.\\d{3})\\)")[,2:4]
  dat <- apply(extracted, 1:2, as.numeric)
  odd = sapply(dat[,1], as.logical)
  even = !odd
  plot(dat[,2][even], dat[,3][even], type = "p", pch = 20, col = rgb(0,0,0.5,0.5), cex = 0.3,
       xlab = expression(zeta), ylab = expression(Delta*S))
  points(dat[,2][odd], dat[,3][odd], type = "p", pch = 20, col = rgb(0.5,0,0,0.5), cex = 0.3)
  grid()
}
