library(baseballr)
library(dplyr)
source("getMinors.R")

# data <- bref_daily_batter("2015-08-01", "2015-10-03") 
# data %>%
#   dplyr::glimpse()

data <- getMinors(3650)
data %>%
  dplyr::glimpse()

