###################################################################### 
rm(list = ls())
set.seed(123)
library(quantmod)
library(moments)
library(tseries) 
library(HiddenMarkov)
library(ggplot2)
library(reshape2)

###################################################################### 
# Obtaining historical stock data for Apple (AAPL) from Yahoo Finance
getSymbols("AAPL", src = "yahoo", from = "2014-01-01", to = "2024-12-31")
head(AAPL)
chartSeries(AAPL, theme = chartTheme("white"))

#AAPL$Returns <- log(AAPL$AAPL.Close) - log(AAPL$AAPL.Open)
stock_prices <- Cl(AAPL)
prices <- as.numeric(stock_prices) # Convert to numeric vector

# Calculating daily log returns
log_returns <- diff(log(prices), lag = 1)
log_returns <- log_returns[!is.na(log_returns)]  # Remove NAs

# explanatory data analysis
summary(log_returns)
sd(log_returns)
skewness(log_returns)
kurtosis(log_returns)

jarque_bera_result <- jarque.bera.test(log_returns)
print(jarque_bera_result) #reject h0, log returns NOT normally distributed

###################################################################### 
## HMM 2 states

nstates <- 2 # number of states

Pi_init <- matrix(1 / nstates, nrow = nstates, ncol = nstates) # transition probabilities (equal probabilities)

delta_init <- rep(1 / nstates, nstates) # initial state probabilities

# emission parameters (mean and variance for Gaussian emissions)
emission_params <- list(mean = c(-0.01, 0.01), sd = c(0.02, 0.05))

# HMM model
hmm_model <- dthmm(x = log_returns, 
                   Pi = Pi_init, 
                   delta = delta_init, 
                   distn = "norm", 
                   pm = emission_params)

fitted_hmm <- BaumWelch(hmm_model)
summary(fitted_hmm)

# fitted model parameters
fitted_hmm$Pi       # Transition matrix
fitted_hmm$delta    # Initial probabilities
fitted_hmm$pm       # Emission parameters (means and variances)

stationary_probabilities <- eigen(t(fitted_hmm$Pi))$vectors[,1]
stationary_probabilities <- stationary_probabilities /sum(stationary_probabilities)

hidden_states <- Viterbi(fitted_hmm)


# Adding hidden states to a data frame
sequence_data <- data.frame(time = 1:length(log_returns), 
                            log_returns = log_returns, 
                            state = factor(hidden_states))

ggplot(sequence_data, aes(x = time, y = log_returns)) +
  geom_line() +
  geom_rect(aes(xmin = time - 0.5, xmax = time + 0.5, ymin = -Inf, ymax = Inf, fill = state), 
            alpha = 0.2, inherit.aes = FALSE) +
  theme_minimal() +
  labs(title = "Log Returns with Regime Shading", fill = "State")

# Adding hidden states to the data for analysis
log_returns_with_states <- data.frame(log_returns = log_returns, 
                                      state = hidden_states)

head(log_returns_with_states)

ggplot(log_returns_with_states, aes(x = 1:length(log_returns), y = log_returns)) +
  geom_line() +
  geom_point(aes(color = factor(state)), size = 1.5) +
  labs(title = "Log Returns with Hidden States", color = "State") +
  theme_minimal()


# log-likelihood from the fitted model
logL <- fitted_hmm$LL
nlogL_2states <- -logL

# Number of parameters
nstates <- length(fitted_hmm$delta)  # Number of states
transition_params <- nstates * (nstates - 1)  # Free parameters in transition matrix
emission_params <- length(unlist(fitted_hmm$pm))  # Emission parameters
initial_params <- nstates - 1  # Initial probabilities
num_params <- transition_params + emission_params + initial_params

# Number of observations
num_obs <- length(log_returns)

# AIC, BIC, HQC calculations
AIC_2states <- -2 * logL + 2 * num_params
BIC_2states <- -2 * logL + num_params * log(num_obs)
HQC_2states <- -2 * logL + 2 * num_params * log(log(num_obs))

# Analysing residuals
residuals <- residuals(fitted_hmm)
hist(residuals, main="HMM: Histogram of Pseudo Residuals")
qqnorm(residuals, main="HMM: Q-Q Plot of Pseudo Residuals")
print(sum(fitted_hmm$delta))
print(fitted_hmm$Pi %*% rep(1, ncol(fitted_hmm$Pi)))

#Plot gaussian mixture on log returns 

state_means <- fitted_hmm$pm$mean
state_sds <-fitted_hmm$pm$sd

###### Plotting Mixture Density with Bell Curve #####
#  Gaussian mixture density calculation
dist1 <- dnorm(log_returns, mean = state_means[1], sd = state_sds[1])
dist2 <- dnorm(log_returns, mean = state_means[2], sd = state_sds[2])

mixture_2_states <- numeric(length(log_returns))

for(i in 1:length(log_returns)){
  mixture_2_states[i] <- stationary_probabilities[1]*dist1[i] + stationary_probabilities[2]*dist2[i] 
}

hist(log_returns, probability = TRUE, col = rgb(0.9, 0.9, 0.9, 0.5), border = "darkgray", 
     xlim = range(log_returns), ylim = c(0, max(c(mixture_2_states, density(log_returns)$y))),
     breaks = 30, main = "2 state HMM: Plot of Gaussian Mixture Density against Log Returns",
     xlab = "Log Returns", ylab = "Density")

points(log_returns, mixture_2_states, col = "lightblue", pch = 16)

curve(dnorm(x, mean = mean(log_returns), sd = sd(log_returns)), col = "gray", lwd = 2, add = TRUE)

legend("topright",  
       legend = c("Gaussian Mixture Density", "Normal Distribution"), 
       col = c("lightblue", "gray"),  
       pch = c(16, NA), 
       lwd = c(NA, 2), 
       bty = "n",  
       cex = 0.8)  
##################################


# Defining Gaussian density function
gaussian_density <- function(x, mean, sd) {
  dnorm(x, mean = mean, sd = sd)
}
# data frame for the density values to overlay
density_data <- data.frame(x = seq(min(log_returns), max(log_returns), length.out = 2000))
for (i in 1:length(state_means)) {
  density_data[[paste0("State", i)]] <- gaussian_density(density_data$x, state_means[i], state_sds[i])
}

density_melted <- melt(density_data, id.vars = "x", variable.name = "State", value.name = "Density")

# Plotting log returns with Gaussian mixture
ggplot() +
  # Histogram of log returns
  geom_histogram(aes(x = log_returns, y = after_stat(density)), bins = 30, fill = "gray", alpha = 0.5) +
  # Overlay Gaussian mixture components
  geom_line(data = density_melted, aes(x = x, y = Density, color = State), linewidth = 1) +
  labs(title = "Log Returns with Gaussian Mixture", x = "Log Returns", y = "Density", color = "State") +
  theme_minimal()

###################################################################### 
## HMM 3 states

nstates <- 3 # number of states

Pi_init <- matrix(1 / nstates, nrow = nstates, ncol = nstates) # transition probabilities (equal probabilities)

delta_init <- rep(1 / nstates, nstates) #initial state probabilities

emission_params <- list(mean = c(-0.01, 0, 0.01), sd = c(0.02, 0.05, 0.03))  # emission parameters (mean and variance for Gaussian emissions)

# HMM model
hmm_model <- dthmm(x = log_returns, 
                   Pi = Pi_init, 
                   delta = delta_init, 
                   distn = "norm", 
                   pm = emission_params)

fitted_hmm <- BaumWelch(hmm_model)
summary(fitted_hmm)

# fitted model parameters
fitted_hmm$Pi       # Transition matrix
fitted_hmm$delta    # Initial probabilities
fitted_hmm$pm       # Emission parameters (means and variances)

stationary_probabilities <- eigen(t(fitted_hmm$Pi))$vectors[,1]
stationary_probabilities <- stationary_probabilities /sum(stationary_probabilities)


hidden_states <- Viterbi(fitted_hmm)

# Adding hidden states to the data for analysis
log_returns_with_states <- data.frame(log_returns = log_returns, 
                                      state = hidden_states)

head(log_returns_with_states)

ggplot(log_returns_with_states, aes(x = 1:length(log_returns), y = log_returns)) +
  geom_line() +
  geom_point(aes(color = factor(state)), size = 1.5) +
  labs(title = "Log Returns with Hidden States", color = "State") +
  theme_minimal()

# adding hidden states to a data frame
sequence_data <- data.frame(time = 1:length(log_returns), 
                            log_returns = log_returns, 
                            state = factor(hidden_states))


ggplot(sequence_data, aes(x = time, y = log_returns)) +
  geom_line() +
  geom_rect(aes(xmin = time - 0.5, xmax = time + 0.5, ymin = -Inf, ymax = Inf, fill = state), 
            alpha = 0.2, inherit.aes = FALSE) +
  scale_fill_manual(values = c("lightblue", "lightpink", "lightgreen")) +
  labs(title = "Log Returns with Regime Shading", fill = "State") +
  theme_minimal()

# log-likelihood from the fitted model
logL <- fitted_hmm$LL
nlogL_3states <- -logL

# Number of parameters
nstates <- length(fitted_hmm$delta)  # Number of states
transition_params <- nstates * (nstates - 1)  # Free parameters in transition matrix
emission_params <- length(unlist(fitted_hmm$pm))  # Emission parameters
initial_params <- nstates - 1  # Initial probabilities
num_params <- transition_params + emission_params + initial_params

# Number of observations
num_obs <- length(log_returns)

# Calculate AIC and BIC
AIC_3states <- -2 * logL + 2 * num_params
BIC_3states <- -2 * logL + num_params * log(num_obs)
HQC_3states <- -2 * logL + 2 * num_params * log(log(num_obs)) 

# Analysing residuals
residuals <- residuals(fitted_hmm)
hist(residuals, main="HMM: Histogram of Pseudo Residuals")
qqnorm(residuals, main="HMM: Q-Q Plot of Pseudo Residuals")
print(sum(fitted_hmm$delta))
print(fitted_hmm$Pi %*% rep(1, ncol(fitted_hmm$Pi)))

# Plot log returns with Gaussian mixture
state_means <- fitted_hmm$pm$mean
state_sds <-fitted_hmm$pm$sd

###### Plotting Mixture Density with Bell Curve #####
#  Gaussian mixture density calculation
dist1 <- dnorm(log_returns, mean = state_means[1], sd = state_sds[1])
dist2 <- dnorm(log_returns, mean = state_means[2], sd = state_sds[2])
dist3 <- dnorm(log_returns, mean = state_means[3], sd = state_sds[3])

mixture_3_states <- numeric(length(log_returns))

for(i in 1:length(log_returns)){
  mixture_3_states[i] <- stationary_probabilities[1]*dist1[i] + stationary_probabilities[2]*dist2[i] + stationary_probabilities[3]*dist3[i] 
}

hist(log_returns, probability = TRUE, col = rgb(0.9, 0.9, 0.9, 0.5), border = "darkgray", 
     xlim = range(log_returns), ylim = c(0, max(c(mixture_3_states, density(log_returns)$y))),
     breaks = 30, main = "3 state HMM: Plot of Gaussian Mixture Density against Log Returns",
     xlab = "Log Returns", ylab = "Density")

points(log_returns, mixture_3_states, col = "lightgreen", pch = 16)

curve(dnorm(x, mean = mean(log_returns), sd = sd(log_returns)), col = "gray", lwd = 2, add = TRUE)

legend("topright",  
       legend = c("Gaussian Mixture Density", "Normal Distribution"),  # Labels 
       col = c("lightgreen", "gray"),
       pch = c(16, NA),  
       lwd = c(NA, 2), 
       bty = "n",  
       cex = 0.8)  
##################################


# Define Gaussian density function
gaussian_density <- function(x, mean, sd) {
  dnorm(x, mean = mean, sd = sd)
}
# Create a data frame for the density values to overlay
density_data <- data.frame(x = seq(min(log_returns), max(log_returns), length.out = 2000))
for (i in 1:length(state_means)) {
  density_data[[paste0("State", i)]] <- gaussian_density(density_data$x, state_means[i], state_sds[i])
}
# Reshape the data for ggplot2
density_melted <- melt(density_data, id.vars = "x", variable.name = "State", value.name = "Density")


ggplot() +
  # Histogram of log returns
  geom_histogram(aes(x = log_returns, y = after_stat(density)), bins = 30, fill = "gray", alpha = 0.5) +
  # Overlay Gaussian mixture components
  geom_line(data = density_melted, aes(x = x, y = Density, color = State), linewidth = 1) +
  labs(title = "Log Returns with Gaussian Mixture", x = "Log Returns", y = "Density", color = "State") +
  theme_minimal()

###################################################################### 
## HMM 4 states

# Define the number of states
nstates <- 4

# Initialize transition probabilities (equal probabilities)
Pi_init <- matrix(1 / nstates, nrow = nstates, ncol = nstates)

delta_init <- rep(1 / nstates, nstates) #  initial state probabilities

emission_params <- list(mean = c(-0.01, 0, 0.01, 0.02), sd = c(0.02, 0.05, 0.03, 0.05))  # emission parameters (mean and variance for Gaussian emissions)

# HMM model
hmm_model <- dthmm(x = log_returns, 
                   Pi = Pi_init, 
                   delta = delta_init, 
                   distn = "norm", 
                   pm = emission_params)

fitted_hmm <- BaumWelch(hmm_model)
summary(fitted_hmm)

# fitted model parameters
fitted_hmm$Pi       # Transition matrix
fitted_hmm$delta    # Initial probabilities
fitted_hmm$pm       # Emission parameters (means and variances)

stationary_probabilities <- eigen(t(fitted_hmm$Pi))$vectors[,1]
stationary_probabilities <- stationary_probabilities /sum(stationary_probabilities)
stationary_probabilities <- Re(stationary_probabilities)

hidden_states <- Viterbi(fitted_hmm)

# Adding the hidden states to the data for analysis
log_returns_with_states <- data.frame(log_returns = log_returns, 
                                      state = hidden_states)

head(log_returns_with_states)


ggplot(log_returns_with_states, aes(x = 1:length(log_returns), y = log_returns)) +
  geom_line() +
  geom_point(aes(color = factor(state)), size = 1.5) +
  labs(title = "Log Returns with Hidden States", color = "State") +
  theme_minimal()


sequence_data <- data.frame(time = 1:length(log_returns), 
                            log_returns = log_returns, 
                            state = factor(hidden_states))

ggplot(sequence_data, aes(x = time, y = log_returns)) +
  geom_line() +
  geom_rect(aes(xmin = time - 0.5, xmax = time + 0.5, ymin = -Inf, ymax = Inf, fill = state), 
            alpha = 0.2, inherit.aes = FALSE) +
  scale_fill_manual(values = c("lightblue", "lightpink", "lightgreen", "lightyellow")) +
  labs(title = "Log Returns with Regime Shading", fill = "State") +
  theme_minimal()


# log-likelihood from the fitted model
logL <- fitted_hmm$LL
nlogL_4states <- -logL

# Number of parameters
nstates <- length(fitted_hmm$delta)  # Number of states
transition_params <- nstates * (nstates - 1)  # Free parameters in transition matrix
emission_params <- length(unlist(fitted_hmm$pm))  # Emission parameters
initial_params <- nstates - 1  # Initial probabilities
num_params <- transition_params + emission_params + initial_params

# Number of observations
num_obs <- length(log_returns)

# Calculate AIC and BIC
AIC_4states <- -2 * logL + 2 * num_params
BIC_4states <- -2 * logL + num_params * log(num_obs)
HQC_4states <- -2 * logL + 2 * num_params * log(log(num_obs)) 


# Analysing residuals
residuals <- residuals(fitted_hmm)
hist(residuals, main="HMM: Histogram of Pseudo Residuals")
qqnorm(residuals, main="HMM: Q-Q Plot of Pseudo Residuals")
print(sum(fitted_hmm$delta))
print(fitted_hmm$Pi %*% rep(1, ncol(fitted_hmm$Pi)))

# Plot log returns with Gaussian mixture
state_means <- fitted_hmm$pm$mean
state_sds <-fitted_hmm$pm$sd


###### Plotting Mixture Density with Bell Curve #####
#  Gaussian mixture density calculation
dist1 <- dnorm(log_returns, mean = state_means[1], sd = state_sds[1])
dist2 <- dnorm(log_returns, mean = state_means[2], sd = state_sds[2])
dist3 <- dnorm(log_returns, mean = state_means[3], sd = state_sds[3])
dist4 <- dnorm(log_returns, mean = state_means[4], sd = state_sds[4])

mixture_4_states <- numeric(length(log_returns))

for(i in 1:length(log_returns)){
  mixture_4_states[i] <- stationary_probabilities[1]*dist1[i] + stationary_probabilities[2]*dist2[i] + stationary_probabilities[3]*dist3[i] + stationary_probabilities[4]*dist4[i] 
}

#plot(log_returns, mixture_4_states, type = "p", col = "lightpink", pch = 16,  
    # xlab = "Log Returns", ylab = "Density", 
    # main = "4 state HMM: Plot of Gaussian Mixture Density against Log Returns", 
    # xlim = range(log_returns), ylim = c(0, max(c(mixture_4_states, density(log_returns)$y))))

#curve(dnorm(x, mean = mean(log_returns), sd = sd(log_returns)), col = "gray", lwd = 2, add = TRUE)
#hist(log_returns, probability = TRUE, col = rgb(0.9, 0.9, 0.9,0.5), border = "darkgray", 
    # add = TRUE, breaks = 30)




hist(log_returns, probability = TRUE, col = rgb(0.9, 0.9, 0.9, 0.5), border = "darkgray", 
     xlim = range(log_returns), ylim = c(0, max(c(mixture_4_states, density(log_returns)$y))),
     breaks = 30, main = "4 state HMM: Plot of Gaussian Mixture Density against Log Returns",
     xlab = "Log Returns", ylab = "Density")

points(log_returns, mixture_4_states, col = "lightpink", pch = 16)

curve(dnorm(x, mean = mean(log_returns), sd = sd(log_returns)), col = "gray", lwd = 2, add = TRUE)

legend("topright",  
       legend = c("Gaussian Mixture Density", "Normal Distribution"),  
       col = c("lightpink", "gray"),  
       pch = c(16, NA),  
       lwd = c(NA, 2), 
       bty = "n", 
       cex = 0.8)  

##################################




# Define Gaussian density function
gaussian_density <- function(x, mean, sd) {
  dnorm(x, mean = mean, sd = sd)
}
# Create a data frame for the density values to overlay
density_data <- data.frame(x = seq(min(log_returns), max(log_returns), length.out = 2000))
for (i in 1:length(state_means)) {
  density_data[[paste0("State", i)]] <- gaussian_density(density_data$x, state_means[i], state_sds[i])
}
# Reshape the data for ggplot2

density_melted <- melt(density_data, id.vars = "x", variable.name = "State", value.name = "Density")

ggplot() +
  # Histogram of log returns
  geom_histogram(aes(x = log_returns, y = after_stat(density)), bins = 30, fill = "gray", alpha = 0.5) +
  # Overlay Gaussian mixture components
  geom_line(data = density_melted, aes(x = x, y = Density, color = State), linewidth = 1) +
  labs(title = "Log Returns with Gaussian Mixture", x = "Log Returns", y = "Density", color = "State") +
  theme_minimal()


print(AIC_2states)
print(AIC_3states)
print(AIC_4states)
print(BIC_2states)
print(BIC_3states)
print(BIC_4states)
print(nlogL_2states)
print(nlogL_3states)
print(nlogL_4states)
print(HQC_2states)
print(HQC_3states)
print(HQC_4states)