library(dplyr)
library(tidyr)

data <- read.csv("Data/common2.csv", header = TRUE)

data <- data %>% select(BA.x, BA.y, OBP.x, OBP.y, SLG.x, SLG.y)

write.csv(data, "Data/common_cleaned.csv", row.names = FALSE)
