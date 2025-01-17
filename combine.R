library(dplyr)
library(tidyr)

major_data <- read.csv("Data/majorData.csv", header = TRUE)
minor_data <- read.csv("Data/minorData.csv", header = TRUE)
minor_data_short <- read.csv("Data/minorShortData.csv", header = TRUE)

major_data <- major_data %>% distinct(Name, .keep_all = TRUE)
minor_data_short <- minor_data_short %>% distinct(Name, .keep_all = TRUE)

# Perform an inner join on the Name and Age columns
common_data <- inner_join(minor_data_short, major_data, by = "Name")

# Display the resulting data
common_data <- common_data %>% select(Name, BA.x, BA.y, OBP.x, OBP.y, SLG.x, SLG.y, OPS.x, OPS.y, PA.x, PA.y)
common_data <- common_data %>% drop_na()
common_data %>% glimpse()

write.csv(common_data, "Data/common2.csv", row.names = FALSE)

