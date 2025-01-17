library(baseballr)
library(dplyr)
library(tidyr)
source("getMinors.R")
source("getMajors.R")
source("minorDaily.R")

# print("Majors:")

# majorData <- getMajors(3650) %>% distinct(Name, Age, .keep_all = TRUE)
# majorData %>% dplyr::glimpse()

print("Minors:")

# minorData1 <- minorDaily("2007-01-01","2016-01-01") %>% distinct(Name, Age, .keep_all = TRUE)
# minorData2 <- minorDaily("2016-01-01","2025-01-01") %>% distinct(Name, Age, .keep_all = TRUE)

# # Full join minorData1 and minorData2
# minorData <- full_join(minorData1, minorData2, by = c("Name", "Age"))

# # Remove duplicates
# minorData <- minorData %>% distinct(Name, Age, .keep_all = TRUE)

# # Display the resulting data
# minorData %>% glimpse()

# minorData1 %>% dplyr::glimpse()

# minorData <- getMinors(4650) %>% distinct(Name, Age, .keep_all = TRUE)
# minorData %>% dplyr::glimpse()

# Perform an inner join on the Name and Age columns
commonData <- inner_join(minorData, majorData, by = "Name")

# Display the resulting data
commonData <- commonData %>% select(Name, BA.x, BA.y, PA.x, PA.y, OBP.x, OBP.y)
commonData <- commonData %>% drop_na()
commonData %>% glimpse()

# write.csv(commonData, "Data/commonData.csv", row.names = FALSE)
# write.csv(majorData, "Data/majorData.csv", row.names = FALSE)
write.csv(minorData, "Data/minorShortData.csv", row.names = FALSE)

# write.csv(majorData, "majorData.csv", row.names = FALSE)

# write.csv(minorData, "minorData.csv", row.names = FALSE)