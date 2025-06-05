#Install and load necessary pacakages for webscraping 
install.packages("rvest")
install.packages("httr")
install.packages("dplyr")
install.packages("stringr")
install.packages("textstem")
install.packages("tidyr")
install.packages("tidytext")
install.packages("topicmodels")
install.packages("ggplot2")
install.packages("cluster")
install.packages("wordcloud")


library(rvest)
library(httr)
library(dplyr)
library(stringr)
library(tm)
library(tidytext)
library(textstem)
library(tidyr)
library(topicmodels)
library(cluster)
library(wordcloud)
library(ggplot2)
library(caret)

# making data frame to hold hospitals and their links for webscraping destination
hospital_data <- data.frame(
  Name = c("Mayo Clinic Hospital", "New York-Presbyterian Hospital", "NYU Langone Health", 
           "Brigham and Women's Hospital", "Massachusetts General Hospital", "Cleveland Clinic", 
           "Northwestern Memorial Hospital", "Rush University Medical Center", 
           "Houston Methodist Hospital", "Johns Hopkins Hospital", 
           "Ronald Reagan UCLA Medical Center", "Cedars-Sinai Medical Center", 
           "UCSF Helen Diller Medical Center", "Stanford Health Care"),
  URL = c("https://www.healthgrades.com/hospital/mayo-clinic-hospital-saint-marys-campus-2b0f2a",
          "https://www.healthgrades.com/hospital/new-york-presbyterian-hospital-1e6252",
          "https://www.healthgrades.com/hospital/nyu-langone-hospitals-22df4b",
          "https://www.healthgrades.com/hospital/brigham-and-womens-hospital-c26308",
          "https://www.healthgrades.com/hospital/massachusetts-general-hospital-60c282",
          "https://www.healthgrades.com/hospital/cleveland-clinic-4a8371",
          "https://www.healthgrades.com/hospital/northwestern-memorial-hospital-f957a8?page=2",
          "https://www.healthgrades.com/hospital/rush-university-medical-center-f0b529",
          "https://www.healthgrades.com/hospital/houston-methodist-hospital-ab4fb4",
          "https://www.healthgrades.com/hospital/the-johns-hopkins-hospital-3390b9",
          "https://www.healthgrades.com/hospital/ronald-reagan-ucla-medical-center-4474ca",
          "https://www.healthgrades.com/hospital/cedars-sinai-medical-center-ad1dca",
          "https://www.healthgrades.com/hospital/ucsf-helen-diller-medical-center-at-parnassus-heights-a88041",
          "https://www.healthgrades.com/hospital/stanford-health-care-tri-valley-07388d")
)

View(hospital_data)
all_text <- ""
for (i in 1:nrow(hospital_data)) {
  url <- hospital_data$URL[i]
  page <- read_html(httr::GET(url))
  hospital_name <- page %>% html_node("h1.hospital-summary-name") %>% html_text()
  comments <- page %>% html_nodes("div.review-comment") %>% html_text()
  full_text <- paste(hospital_name, comments, sep=" | ")
  all_text <- paste0(all_text, "\n\n", full_text)
}

#writing all reviews to a text file to be made to a dataframe 
reviews <- readLines("hospital_reviews.txt")
# Convert to a data frame with separate columns
HR_DF <- data.frame(
  text = reviews,
  stringsAsFactors = FALSE
) %>%
  separate(text, into = c("hospital", "review"), sep = " \\| ", extra = "merge")
#remove na values 
HR_DF <- na.omit(HR_DF)

#preproceesing the review columns
HR_DF$review <- tolower(HR_DF$review)
HR_DF$review <- stripWhitespace(HR_DF$review)


#sentiment Analysis using Bing and Adinn lexico

bing <- get_sentiments("bing")
afinn <- get_sentiments("afinn")
custom_stopwords <- c("hospital", "clinic", "doctor", "patient", "medical", "health", "center", "Doctors", "experience")
custom_stopwords <- c(custom_stopwords, stopwords("en"))
custom_stopwords_df <- data.frame(word = custom_stopwords)


HR_Bing <- HR_DF %>%
  unnest_tokens(word, review) %>%
  anti_join(custom_stopwords_df) %>%
  inner_join(bing, by = "word") 

HR_Afinn  <- HR_DF %>%
  unnest_tokens(word, review) %>%
  anti_join(custom_stopwords_df) %>%
  inner_join(afinn, by = "word") 

summary(HR_DF)
HR_DF$hospital <- as.factor(HR_DF$hospital)
HR_DF$review <- as.character(HR_DF$review)



#Visualizations for Sentiment Analysis
library(ggplot2)

#boxplot for posiive vs negative reviews out of all the hospitals reviewed
ggplot(HR_Bing, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Sentiment Analysis of Hospital Reviews (Bing Lexicon)",
       x = "Sentiment",
       y = "Word Count",
       fill = "Sentiment")


#compares positive vs negative sentiments across each hospital
ggplot(HR_Bing, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  facet_wrap(~hospital) +
  theme_minimal() +
  labs(title = "Sentiment Comparison Across Hospitals",
       x = "Sentiment",
       y = "Word Count",
       fill = "Sentiment")


#Wordcloud for postive and negative sentimments 
HR_Positive <- HR_Bing %>% filter(sentiment == "positive")
HR_Negative <- HR_Bing %>% filter(sentiment == "negative")


#positive word cloud
word_freq_positive <- HR_Positive %>%
  count(word) 
wordcloud(words = word_freq_positive$word, freq = word_freq_positive$n,
          min.freq = 2, max.words = 100, scale = c(0.75, 1), random.order = FALSE)

#negative word cloud
word_freq_negative <- HR_Negative %>%
  count(word) 
wordcloud(words = word_freq_negative$word, freq = word_freq_negative$n,
          min.freq = 1, max.words = 100, scale = c(3, 1.0), 
          random.order = FALSE)



# Convert text data into a corpus for LDA

source_data <- VectorSource(HR_DF$review)
HRCorpus <- VCorpus(source_data) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, custom_stopwords) %>%
  tm_map(stripWhitespace)

HR_DTM <- DocumentTermMatrix(HRCorpus)

empty_docs <- which(rowSums(as.matrix(HR_DTM)) == 0)
HR_DTM <- HR_DTM[rowSums(as.matrix(HR_DTM)) > 0, ]
# Apply LDA with 5 topics
HR_DTM <- removeSparseTerms(HR_DTM, 0.99)
HR_LDA <- LDA(HR_DTM, k = 5)

HR_topics <- terms(HR_LDA, 6)
print(HR_topics)

View(HR_topics)
HR_LDA_gamma <- tidy(HR_LDA, matrix = "gamma")

#barchart of topic distribution

ggplot(HR_LDA_gamma, aes(x = factor(topic), y = gamma, fill = factor(topic))) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Topic Distribution in Hospital Reviews", x = "Topic", y = "Probability", fill = "Topic")

#heatmap of topic distribution

ggplot(HR_LDA_gamma, aes(x = document, y = factor(topic), fill = gamma)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  labs(title = "Heatmap of Topic Distribution", x = "Document", y = "Topic")

#Topic distribution per review

HR_LDA_gamma <- tidy(HR_LDA, matrix = "gamma")
empty_docs <- which(rowSums(as.matrix(HR_DTM)) == 0)

ggplot(HR_LDA_gamma, aes(x = factor(topic), y = gamma, fill = factor(topic))) +
  geom_bar(stat = "identity") +
  facet_wrap(~ document) +
  theme_minimal() +
  labs(title = "Topic Distribution Per Document", x = "Topic", y = "Probability", fill = "Topic")


#Steps to start predictive modeling 
# Add sentiment scores to each dataset
HR_Positive$sentiment_score <- 1
HR_Negative$sentiment_score <- 0

# Combine both datasets into a single DataFrame
HR_Sentiment <- bind_rows(HR_Positive, HR_Negative)

#grouping by hospital to get sentiment score
# Summarize sentiment scores per hospital
HR_Hospital_Sentiment <- HR_Sentiment %>%
  group_by(hospital) %>%
  summarise(sentiment_score_avg = mean(sentiment_score),
            sentiment_count = n())

# View summary
print(HR_Hospital_Sentiment)



#Predictive Modeling
library(caret)
# creation of model for predicting sentiment score based on words
predict_model <- glm(sentiment_score ~ word, data = HR_Sentiment, family = "binomial")
summary(predict_model)


#application of model to estimate sentiment trends
HR_Sentiment$predicted_sentiment <- predict(predict_model, type = "response")
# aggregate predictions per hospital
HR_Hospital_Predictions <- HR_Sentiment %>%
  group_by(hospital) %>%
  summarise(predicted_sentiment_avg = mean(predicted_sentiment))

#barchart of sentiment trends per hospital
ggplot(HR_Hospital_Predictions, aes(x = hospital, y = predicted_sentiment_avg, fill = hospital)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Predicted Sentiment Trends Across Hospitals", x = "Hospital", y = "Predicted Sentiment Score", fill = "Hospital") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

