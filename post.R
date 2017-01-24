library(h5)
library(ggplot2)

## Utility function to read a HDF5 file and create a data frame with it.
read.h5 <- function(filename) {

  file <- h5file(filename,"r")
  dset <- file["means"]
  dat <- data.frame(β = c(dset[,1]), mean = c(dset[,2]), t = c(dset[,3]))
  dat
}

print.stats <- function(filename) {

  dat <- read.h5(filename)
  print(sprintf("t  = %f", mean(dat$t)))
  print(sprintf("SD = %f", sd(dat$t)))
}

## Plots each field mean value for each temperature as a dot.
mean.scatterplot <- function(filename) {

  dat <- read.h5(filename)

  graphics.off()
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
  graphics.off()
  dev.new(width = 4.5, height = 3.5)
  par(mar = c(4,5,1.5,1.5) + 0.1)
  breaks = 16*(1:5)
  ggplot(data = dat, aes(x = Ni, y = t)) + geom_point() +
         geom_errorbar(aes(ymin = t-sd, ymax = t+sd), width=0.01) +
         scale_x_log10(expression(N[i]), breaks = breaks) + scale_y_log10("t [s]")
}

efficiency <- function(file, outfile = "") {

  dat <- rbind(read.table(file, header = TRUE))
  dat$eff <- with(dat, N0*Ni^3 / t)
  dat$eff <- dat$eff / max(dat$eff)

  graphics.off()
  dev.new(width = 5.5, height = 3.5)
  par(mar = c(4,5,1.5,1.5) + 0.1)
  breaks = 16*(1:dim(dat)[1])
  my.labels <- c("No subdivision","4D subdomains","Shared memory")
  plt <- ggplot(data = dat, aes(x = Ni, y = eff, color = factor(Run, labels = my.labels))) +
         geom_point() + labs(color = "") +
         scale_x_log10(expression(N[i]), breaks = breaks) +
         scale_y_continuous(expression(epsilon), limits = c(0, max(dat$eff))) +
         theme(axis.title.y = element_text(angle = 0))

  print(plt)

  if (outfile != "") {
    dev.print(pdf, width = 5.5, height = 3.5, file = outfile)
  }
}
