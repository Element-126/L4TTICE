library(h5)
library(ggplot2)

vac.scatterplot <- function(filename) {

  file <- h5file(filename,"r")
  dset <- file["means"]
  dat <- data.frame(β = c(dset[,1]), mean = c(dset[,2]), t = c(dset[,3]))

  dev.new(width = 5.5, height = 4.5)
  par(mar = c(4,5,1.5,1.5) + 0.1)
  breaks = c(1/4, 5/16, 3/8, 1/2 , 5/8, 3/4, 1.0, 1.25, 1.5, 2.0, 2.5, 3, 4, 5, 6, 8, 10)
  ggplot(dat, aes(x = β, y = mean)) + geom_point(alpha = 0.2) +
         scale_x_log10(expression(beta), breaks = breaks, minor_breaks = NULL) +
         theme(axis.text.x = element_text(angle = -45, vjust = 0.5)) +
         ylab(expression(symbol("\341")~Phi[H]~symbol("\361")[list(tau,x)]))
}
