##########################################################
# Step 0: Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(lubridate)
library(ggplot2) 
library(scales)
library(gridExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
options("digits" = 5)
options(pillar.sigfig = 5) # Displays 4 significant digits
options(timeout = 120)
set.seed(1, sample.kind = "Rounding")  # R >= 3.6

##########################################################
# Step 0: Load MovieLens 10M locally (download if missing)
##########################################################

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Persist splits (useful for the Rmd and for reproducibility)
saveRDS(edx, "edx.rds")
saveRDS(final_holdout_test, "final_holdout_test.rds")

##########################################################
# Step 1: Create Train and Test Sets from edx
# Goal: Tune the model parameters without data leakage
##########################################################
set.seed(1, sample.kind = "Rounding")

# I use the 'edx' dataset created in the previous step.

# Split 'edx' into 2 parts:
# - train_set (90%): To teach the algorithms
# - test_set(10%): To test the RMSE while developing
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE )

train_set <- edx[-test_index,]
temp_test_set <- edx[test_index,]

# Data Cleaning step
# Ensure that only test on users and movies that actually exist in the training set.
# If a movie is in the test set but was never seen in the training set , prediction is not possible wih this method.

test_set <- temp_test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add the rows removed during the cleaning back into the training set
# so we do not lose any data
removed <- anti_join(temp_test_set, test_set)
train_set <- rbind(train_set, removed)

# Clean up temporary files to save memory
rm(test_index, temp_test_set, removed)

# Verification
cat("Internal Train set rows: ", nrow(train_set), "\n")
cat("Internal Test set rows: ", nrow(test_set), "\n")


##########################################################
# EDA: 1. Preparation of Date Variable (Review Date)
# Only prepares the date for the b_t visualization
##########################################################
edx <- edx %>%
  mutate(review_date = round_date(as_datetime(timestamp), unit = "week"))

# Define mu_hat to use in visualizations
mu_hat <- mean(edx$rating)


##########################################################
# Step 2: Basic Models (Mean + Effect)
##########################################################

# 0. Define the RMSE function (The metrtic for success)
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# ---- Model 1: Simple Average (Just the mean) ----
# Predict the every single user gives the average rating

#naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse <- RMSE(test_set$rating, rep(mu_hat, nrow(test_set)))


# Save the result in a table (So keep adding rows to this table)
rmse_results <- tibble(method = "Just the Average", RMSE = naive_rmse)

print(rmse_results)
# Expectation: RMSe around 1.06 (Thjs is out baseline, so I must beat this)

# --- Model 2: Movie Effect (b_i) ---
# Calculate the deviation of each movie from the average
# b_i = average (rating - mu) for each movieId
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

# Predict on the test set: mu + b_i
predicted_ratings_bi <- test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  mutate(pred = mu_hat + coalesce(b_i, 0)) %>%
  pull(pred)

# Calculate RMSE for Model 2
model_2_rmse <- RMSE (test_set$rating, predicted_ratings_bi)

# Add result to table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie Effect Model",
                                 RMSE = model_2_rmse))
# Show progress
print(rmse_results)
# Expectation: RMSe should drop significantly (around 0.94) 


##########################################################
# Step 3: User Effect (b_u)
##########################################################

# --- Model 3: Movie Effect + User Effect  ---
# Calculate the bias for each user (b_u)
# calculate the average of the residuals: (rating - mu - b_i)

user_avgs <- train_set %>%
  left_join (movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Predict on test set: mu + b_i + b_u
predicted_ratings_bu <- test_set %>% 
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu_hat + coalesce(b_i, 0) + coalesce(b_u, 0)) %>%
  pull(pred)

# Calculate RMSE for Model 3
model_3_rmse <- RMSE(test_set$rating, predicted_ratings_bu)

# Add result to table

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effect Model",
                                 RMSE = model_3_rmse))

# Show Progress
print(rmse_results)


##########################################################
# Step 4: Genre Effect Model (b_g)
# Goal: capture bias based on genre (Dramas vs Comedies)
##########################################################

# ---- Model 4: Movie + User + Genre Effect ----

# Calculate bias for each genre combination (b_g)
# Formula: mean(rating - mu - b_i - b_u)
genre_avgs <- train_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))

# Predict on test set
predicted_ratings_bg <- test_set %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  left_join(genre_avgs, by = 'genres') %>%
  mutate(pred = mu_hat + coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0)) %>%
  pull(pred)

# Calculate RMSE for model 4
model_4_rmse <- RMSE (test_set$rating, predicted_ratings_bg)

# Update results table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User + Genre Effect Model",
                          RMSE = model_4_rmse))

# Show progress
print(rmse_results)


##########################################################
# Step 6: Time Effect Model (b_t)
# Goal: capture bias based on time 
##########################################################
# ---- Model 5: Movie + User + Genre + Tme  Effect ----

# --- 1. Create a date column for b_T ---
# convert timestamp to date/time format and round to the nearest week for better grouping
train_set <- train_set %>% mutate(date = as_datetime(timestamp))
test_set  <- test_set %>% mutate(date = as_datetime(timestamp))


# Round date to the nearest week for the time effect (b_t)
train_set <- train_set %>% mutate(week = round_date(date, unit = "week"))
test_set <- test_set %>% mutate(week = round_date(date, unit = "week"))

# --- 2. Calculate the Time effect (b_t) ---
# b_t = average(rating - mu - b_i - b_u - b_g) for each week

time_avgs <- train_set %>%
  left_join(movie_avgs, by =  "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  group_by(week) %>%
  summarize(b_t = mean(rating - mu_hat - b_i - b_u - b_g))

# --- Predict and Calculate RMSE for model 5
predict_ratings_bt <- test_set %>%
  left_join(movie_avgs, by =  "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  left_join(time_avgs, by = "week") %>%
  mutate(pred = mu_hat + coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0) + coalesce(b_t,0)) %>%
  pull(pred)
  
# Calculate RMSE for model 4
model_5_rmse <- RMSE (test_set$rating, predict_ratings_bt)

# Update results table
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie + User + Genre + Time Effect Model",
                                 RMSE = model_5_rmse))


# Show progress
print(rmse_results)


##########################################################
# Step 6: Regularization (Penalized Least Squares)
# Goal: Penalize large estimates formed by small sample sizes
##########################################################

# Use sapply to run a function for each lambda value
eval_rmse_for_lambda <- function(l) {
    # Calculate average (mu)
    mu <- mean(train_set$rating)
    
    # Regularized Movie Effect (b_i)
    # Notice the denominator: n() + l
    b_i <- train_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + l))
    
    # Regularized User Effect (b_u)
    b_u <- train_set %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + l))
    
    # Regularized Genre Effect (b_g)
    b_g <- train_set %>%
      left_join(b_i , by = 'movieId') %>%
      left_join(b_u, by = 'userId') %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - b_i - b_u - mu) / (n() + l)) 
    
    # Regularized Time Effect (b_t)
    b_t <- train_set %>%
      left_join(b_i , by = 'movieId') %>%
      left_join(b_u, by = 'userId') %>%
      left_join(b_g, by = 'genres') %>%
      group_by(week) %>%
      summarize(b_t = sum(rating - b_i - b_u - b_g - mu) / (n() + l)) 
    
    # Predict on internal test set
    predicted_ratings <- test_set %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_g, by = "genres") %>%
      left_join(b_t, by = "week") %>%
      mutate(pred = mu + b_i + b_u + b_g + b_t) %>%
      pull(pred)
    
    # Return RMSE for this lambda
    return(RMSE(test_set$rating, predicted_ratings))
  }
  
  # Define a sequeance of lambdas to test (from 0 to 10 with 0.25 steps)
  lambdas_coarse <- seq (0, 10, 0.25)
  rmses_coarse   <- sapply(lambdas_coarse, eval_rmse_for_lambda)
  lambda_coarse <- lambdas_coarse[which.min(rmses_coarse)]

  # Try to fine tune the lambda ( 0.05 steps)
  
  lambdas_fine <- seq(max(0, lambda_coarse - 0.5), lambda_coarse + 0.5, by = 0.05)
  rmses_fine <- sapply(lambdas_fine, eval_rmse_for_lambda)
  lambda_star <- lambdas_fine[which.min(rmses_fine)]
  rmse_star    <- min(rmses_fine)
  
  cat(sprintf("Best lambda*: %.3f | RMSE (internal test): %.5f\n", lambda_star, rmse_star))
  
  # Show progress
  rmse_results <- bind_rows(rmse_results,
                            tibble(method = "Regularized Movie + User + Genre + Time Effect Model",
                                   RMSE = rmse_star))
  
  print(rmse_results)                          
  

##########################################################
# Step 7: Final Validation (Official RMSE)
##########################################################

# Use the optimal lambda found in Step 6
lambda_final <- lambda_star

# NOTE: Create the time columns (week) on both sets (Edx was created on the EDA) before calculating b_t
#edx <- edx %>% mutate(review_date = round_date(as_datetime(timestamp), unit = "week"))

final_holdout_test <- final_holdout_test %>% mutate(review_date = round_date(as_datetime(timestamp), unit = "week"))

# --- 1. Re-train with the full edx set
# This means recalculating all biases using larger 'edx'
mu_final <- mean(edx$rating)

# Final Regularized Movie Effect (b_i)
b_i_final <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_final) / (n() + lambda_final), .groups = "drop")


# Regularized User Effect (b_u)
b_u_final <- edx %>%
  left_join(b_i_final, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_final) / (n() + lambda_final), .groups = "drop")


# Regularized Genre Effect (b_g)
b_g_final <- edx %>%
  left_join(b_i_final , by = 'movieId') %>%
  left_join(b_u_final, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu_final) / (n() + lambda_final), .groups = "drop") 

# Regularized Time Effect (b_t)
b_t_final <- edx %>%
  left_join(b_i_final , by = 'movieId') %>%
  left_join(b_u_final, by = 'userId') %>%
  left_join(b_g_final, by = 'genres') %>%
  group_by(review_date) %>%
  summarize(b_t = sum(rating - b_i - b_u - b_g - mu_final) / (n() + lambda_final), .groups = "drop") 

# Predict and Evaluate on 'final_holdout_test
predicted_ratings_final <- final_holdout_test %>%
  left_join(b_i_final, by = "movieId") %>%
  left_join(b_u_final, by = "userId") %>%
  left_join(b_g_final, by = "genres") %>%
  left_join(b_t_final, by = "review_date") %>%
  mutate(pred = mu_final +  coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0) + coalesce(b_t,0)) %>%
  pull(pred)

stopifnot(!any(is.na(predicted_ratings_final)))

# Calculate the official RMSE
final_rmse = RMSE(final_holdout_test$rating, predicted_ratings_final)

# Report the official result
cat(sprintf("Offical Validation RMSE = %.5f\n", final_rmse))

# Output files
pred_out <- final_holdout_test %>%
  select(userId, movieId) %>%
  mutate(rating_pred = predicted_ratings_final)

if (!dir.exists("outputs")) dir.create("outputs")
data.table::fwrite(pred_out, "outputs/predictions_final.csv")
writeLines(sprintf("RMSE_official,%.5f", final_rmse), con = "outputs/rmse_official.txt")


# Table evolutive RMSEs  + final
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Official (final_holdout_test)",
                                 RMSE   = final_rmse))
saveRDS(rmse_results, "outputs/rmse_summary.rds")

print(rmse_results %>% arrange(RMSE))


