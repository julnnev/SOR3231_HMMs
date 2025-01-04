rm(list = ls())
set.seed(123)
library(HiddenMarkov)
library(readr)

# Preparing dataset for HMM analysis
data <- read_csv("Desktop/Time Series 2 Assignment/data.csv")
data$`Open Timestamp` <- as.POSIXct(data$`Open Timestamp`, format = "%Y.%m.%d %H:%M:%S")
data$`Open Timestamp` <- as.Date(data$`Open Timestamp`)

data$log_returns <- log(data$Close) - log(data$Open)
# thresholds for bull and bear markets
threshold_positive <- 0.0001  # Small positive return -  bull market
threshold_negative <- -0.0001  # Small negative return - bear market

# labelling the market conditions
data$market_conditions <- ifelse(data$log_returns > threshold_positive, "Bull", 
                            ifelse(data$log_returns < threshold_negative, "Bear", "Neutral"))

nStates <- 3  # Number of hidden states
Pi <- matrix(c(0.7, 0.2, 0.1,    # From Bull (State 1)
               0.3, 0.4, 0.3,    # From Neutral (State 2)
               0.2, 0.3, 0.5),   # From Bear (State 3)
             byrow = TRUE, nrow = 3)
delta <- c(1/3, 1/3, 1/3) # no strong prior belief about the market conditions at the start (equally likely to be in any of the states)

# Assuming normal distributions for each state
# Assuming data$log_returns is the log returns and data$state has the states
bull_returns <- data$log_returns[data$market_conditions == "Bull"]
neutral_returns <- data$log_returns[data$market_conditions == "Neutral"]
bear_returns <- data$log_returns[data$market_conditions == "Bear"]

# Calculate the mean and standard deviation for each state
mean_bull <- mean(bull_returns, na.rm = TRUE)
mean_neutral <- mean(neutral_returns, na.rm = TRUE)
mean_bear <- mean(bear_returns, na.rm = TRUE)

sd_bull <- sd(bull_returns, na.rm = TRUE)
sd_neutral <- sd(neutral_returns, na.rm = TRUE)
sd_bear <- sd(bear_returns, na.rm = TRUE)

# Set the means and standard deviations for the HMM
means <- c(mean_bull, mean_neutral, mean_bear)
sds <- c(sd_bull, sd_neutral, sd_bear)

mod <- dthmm(data$log_returns, 
             Pi = Pi,         # Transition matrix
             delta = delta,   # Initial state distribution
             "norm",          # Normal emission distribution
             list(mean = means, sd = sds))

fitted.mod <- BaumWelch(mod) 

summary(fitted.mod)

fitted.mod$u  #state probabilities

#data$viterbi_states <- Viterbi(fitted.mod)
#state_labels <- c("Bull", "Neutral", "Bear")
#data$decoded_state <- state_labels[data$viterbi_states]
#conf_matrix <- table(Predicted = data$decoded_state, Actual = data$market_conditions)
#print(conf_matrix)
#accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
#print(accuracy)


residuals <- residuals(fitted.mod)


####
hist(residuals, main="HMM: Boxplot of Pseudo Residuals")
qqnorm(residuals, main="HMM: Q-Q Plot of Pseudo Residuals")

print(logLik(fitted.mod))

print(sum(fitted.mod$delta))
print(fitted.mod$Pi %*% rep(1, ncol(fitted.mod$Pi)))

# Decode states using Viterbi algorithm
states <- Viterbi(mod)

# Compare predicted states with actual states (assuming `data$decoded_state` contains actual labels)
comparison_df <- data.frame(Time = 1:length(states),
                            Actual = data$decoded_state,
                            Predicted = factor(states, levels = 1:3, labels = c("Low", "Neutral", "High")))

# Plot the log returns
plot(1:length(data$log_returns), data$log_returns, type = "l", xlab = "Time", ylab = "Log Returns", col = "black")
plot(1:30, data$log_returns[1:30], type = "l", xlab = "Time", ylab = "Log Returns", col = "black")


# Add points for actual states (e.g., "Low" -> blue, "Neutral" -> green, "High" -> red)
points((1:length(data$log_returns))[comparison_df$Actual == "Low"], data$log_returns[comparison_df$Actual == "Low"], col = "blue", pch = 15)
points((1:length(data$log_returns))[comparison_df$Actual == "Neutral"], data$log_returns[comparison_df$Actual == "Neutral"], col = "green", pch = 15)
points((1:length(data$log_returns))[comparison_df$Actual == "High"], data$log_returns[comparison_df$Actual == "High"], col = "red", pch = 15)

# Add points for predicted states
points((1:length(data$log_returns))[comparison_df$Predicted == "Low"], data$log_returns[comparison_df$Predicted == "Low"], col = "blue", pch = 1)
points((1:length(data$log_returns))[comparison_df$Predicted == "Neutral"], data$log_returns[comparison_df$Predicted == "Neutral"], col = "green", pch = 1)
points((1:length(data$log_returns))[comparison_df$Predicted == "High"], data$log_returns[comparison_df$Predicted == "High"], col = "red", pch = 1)

# Optionally, you can highlight the points where the prediction is wrong
wrong <- comparison_df$Actual != comparison_df$Predicted
points((1:length(data$log_returns))[wrong], data$log_returns[wrong], pch = 1, cex = 1.5, col = "black")


