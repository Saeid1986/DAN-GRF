####################################################################
#################     DEEP AUTOENCODER NETWORK     #################
#################          CONNECTED TO            #################
#################    GEOGRAPHICAL RANDOM FOREST    #################
####################################################################


# Developers: Zeinab SOLTANI & Saeid ESMAEILOGHLI
# December 10, 2023
# -----------------------------
# Hardware required: a computer with a processor of 11th Generation
# Intel(R) Core(TM) i5 @ 2.40 GHz, four cores, eight logical
# processors, installed physical memory (RAM) of 8.00 GB, an NVIDIA
# GeForce MX330 graphics card, or higher.
# Software required: R 4.1.2 or higher.
# Program language: R language environment.
# Program size: 17.8 KB.


# ----------------------------- Inputs -----------------------------

setwd("C:/Users/ASUS/Desktop/GitHub")

data <- read.table("C:/Users/ASUS/Desktop/GitHub/Inputs (Test Data)/data.txt",
                   header = TRUE,
                   sep = ""
                   )

# --------------------------- Main Script --------------------------

# A. Load packages

library(compositions)
library(ggplot2)
library(GGally)
library(h2o)
library(RMThreshold)
library(SpatialML)
library(randomForest)


# B. Initialize H2O session

h2o.no_progress()
h2o.init()


# C. Define geographical coordinates and geochemical variables

Coords <- data[ , c(1, 2)]
Geodata <- data[ , c(-1, -2)]


# D. Define plot properties

cor_func <- function(data, mapping, method, symbol, ...){
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  corr <- cor(x, y, method = method, use = "complete.obs")
  colFn <- colorRampPalette(c("blue", "white", "brown1"),
                            interpolate = "spline")
  fill <- colFn(100)[findInterval(corr, seq(-1, 1, length = 100))]
  
  ggally_text(
    label = paste(symbol, as.character(round(corr, 2))),
    mapping = aes(),
    xP = 0.5, yP = 0.5,
    color = "black",
    ...
    ) +
    theme(panel.background = element_rect(fill = fill))
  }

mytheme = theme(strip.background = element_rect(fill = "white"),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank()
                )


# E. Plot pair plots of original data

p_o <- ggpairs(
  Geodata,
  upper = list(continuous = wrap(cor_func,
                                 method = "pearson",
                                 symbol = expression("\u03C1 ="))),
  diag = list(continuous = function(data, mapping, ...) {
    ggally_densityDiag(data = data, mapping = mapping) + 
      theme(panel.background = element_blank())}
    ))

p_o + mytheme


# F. Compositional data analysis (CoDA)

# F.1. Do not perform a given transformation

x <- Geodata

# F.2. Transform original data into clr-transformed data

x_clr <- clr(Geodata)

x_clr <- as.data.frame(x_clr)

# F.3. Transform original data into ilr-transformed data

x_ilr <- ilr(Geodata, V = ilrBase(Geodata))

x_ilr <- as.data.frame(x_ilr)

# F.4. Save CoDA outputs

write.table(x_ilr,
            file = "ILR-Transformed Data.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )


# G. Plot pair plots of CoDA-transformed data

p_t <- ggpairs(
  x_ilr,
  upper = list(continuous = wrap(cor_func,
                                 method = "pearson",
                                 symbol = expression("\u03C1 ="))),
  diag = list(continuous = function(data, mapping, ...) {
    ggally_densityDiag(data = data, mapping = mapping) + 
      theme(panel.background = element_blank())}
  ))

p_t + mytheme


# H. Rescale input signals into range [0,1]

minMax <- function(f) {
  (f - min(f)) / (max(f) - min(f))
  }

y <- as.data.frame(lapply(x_ilr, minMax))


# I. Convert signals to H2O input data

features <- as.h2o(y)


# J. Implement deep autoencoder network (DAN)

sae <- h2o.deeplearning(
       x = seq_along(features),   # A vector containing the names or
                                  # indices of variables to use in
                                  # building the model.
       training_frame = features, # Id of the training data frame.
       weights_column = NULL,     # Column with observation weights.
       pretrained_autoencoder = NULL,
                                  # Pretrained autoencoder model.
       standardize = FALSE,       # Standardize the data.
       activation = "Tanh",       # Activation function. Must be one
                                  # of: "Tanh", "TanhWithDropout",
                                  # "Rectifier",
                                  # "RectifierWithDropout",
                                  # "Maxout", or
                                  # "MaxoutWithDropout".
       hidden = c(24, 18, 12, 6, 12, 18, 24),
                                  # Hidden layer sizes.
       epochs = 100,              # Number of epochs.
       adaptive_rate = TRUE,      # Adaptive learning rate.
       rho = 0.99,                # Adaptive learning rate time
                                  # decay factor.
       epsilon = 1e-08,           # Adaptive learning rate smoothing
                                  # factor.
       rate = 0.005,              # Learning rate.
       rate_annealing = 1e-06,    # Learning rate annealing.
       rate_decay = 1,            # Learning rate decay factor
                                  # between layers.
       momentum_start = 0,        # Initial momentum at the
                                  # beginning of training (try 0.5).
       momentum_ramp = 1e+06,     # Number of training samples for
                                  # which momentum increases.
       momentum_stable = 0,       # Final momentum after the ramp is
                                  # over (try 0.99).
       l1 = 0,                    # L1 regularization.
       l2 = 0,                    # L2 regularization.
       initial_weight_distribution = "UniformAdaptive",
                                  # Initial weight distribution.
                                  # Must be one of:
                                  # "UniformAdaptive", "Uniform", or
                                  # "Normal".
       loss = "Automatic",        # Loss function. Must be one of:
                                  # "Automatic", "Quadratic",
                                  # "Huber", "Absolute", or
                                  # "Quantile".
       distribution = "AUTO",     # Distribution function. Must be
                                  # one of: "AUTO", "bernoulli",
                                  # "multinomial", "gaussian",
                                  # "poisson", "gamma", "tweedie",
                                  # "laplace", "quantile", or
                                  # "huber".
       stopping_rounds = 5,       # Early stopping based on
                                  # convergence of stopping_metric.
       stopping_metric = "MSE",   # Metric to use for early
                                  # stopping. Must be one of: "AUTO"
                                  # or "MSE".
       stopping_tolerance = 0,    # Relative tolerance for
                                  # metric-based stopping criterion.
       fast_mode = FALSE,         # Enable fast mode (minor
                                  # approximation in
                                  # back-propagation).
       variable_importances = TRUE,
                                  # Compute variable importances for
                                  # input features (Gedeon method).
       autoencoder = TRUE,        # Autoencoder.
       sparse = FALSE,            # Sparse data handling.
       average_activation = 0,    # Average activation for sparse
                                  # autoencoder.
       sparsity_beta = 0,         # Sparsity regularization.
       mini_batch_size = 32,      # Mini-batch size.
       verbose = FALSE            # Print scoring history to the
                                  # console.
       )

sae@model[["model_summary"]]
sae@model[["scoring_history"]][["training_mse"]]


# K. Extract deep feature codings (DFCs)

df_codings <- h2o.deepfeatures(
  object = sae,
  data = features,
  layer = 4
  )

df_codings <- as.data.frame(df_codings)

write.table(df_codings,
            file = "Deep Features.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )


# L. Reconstruct data

reconstruction <- predict(
  object = sae,
  newdata = features
  )

reconstruction <- as.data.frame(reconstruction)

write.table(reconstruction,
            file = "Reconstruction.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )

# M. Calculate reconstruction errors (anomaly scores)

anomaly <- h2o.anomaly(
  object = sae,
  data = features,
  per_feature = FALSE
  )

anomaly <- as.data.frame(anomaly)

anomaly_normal <- as.data.frame(lapply(anomaly, minMax))

write.table(anomaly,
            file = "Anomaly Score.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )

write.table(anomaly_normal,
            file = "Normalized Anomaly Score.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )


# N. Combine geochemical variables and deep feature codings

grf_data <- cbind(Geodata, df_codings)


# O. Standardize (zero-mean & unit-variance) GRF-related variables

grf_scale_data <- scale(
  x = grf_data,
  center = TRUE,
  scale = TRUE
  )

grf_scale_data <- as.data.frame(grf_scale_data)


# P. Optimal bandwidth selection

obw <- grf.bw(
  Au ~ DF.L4.C1 + DF.L4.C2 + DF.L4.C3 + DF.L4.C4 + DF.L4.C5 + DF.L4.C6,
                                  # The local model to be fitted.
  dataset = grf_scale_data,       # Numeric data frame of variables.
  kernel = "adaptive",            # The kernel to be used in the
                                  # regression. Options are
                                  # "adaptive" (default) or "fixed".
  coords = Coords,                # A numeric matrix or data frame
                                  # of two columns giving the X,Y
                                  # coordinates of the observations.
  bw.min = 20,                    # Minimum bandwidth that
                                  # evaluation starts.
  bw.max = 25,                    # Maximum bandwidth that
                                  # evaluation ends.
  step = 1,                       # Step for each iteration of the
                                  # evaluation between the min and
                                  # the max bandwidth.
  trees = 10,                     # Number of trees to grow for each
                                  # of the local random forests.
  mtry = NULL,                    # Number of variables randomly
                                  # sampled as candidates at each
                                  # split. The default value is p/3,
                                  # where p is number of variables
                                  # in the formula.
  importance = "impurity",        # Feature importance of the
                                  # dependent variables used as
                                  # input at the random forest.
                                  # Default value is "impurity"
                                  # which refers to the Gini index
                                  # for classification and the
                                  # variance of the responses for
                                  # regression.
  nthreads = 1,                   # Number of threads. Default is
                                  # number of CPUs available.
  forests = FALSE,                # An option to save and export
                                  # (TRUE) or not (FALSE) all the
                                  # local forests.
  weighted = TRUE                 # If TRUE the algorithm calculates
                                  # GRF using the case.weights
                                  # option of the package ranger. If
                                  # FALSE it will calculate local
                                  # random forests without weighting
                                  # each observation in the local
                                  # dataset.
  )

obw[["Best.BW"]]

bandwidths <- obw[["tested.bandwidths"]][["Bandwidth"]]
local_R2 <- obw[["tested.bandwidths"]][["Local"]]

bw_grf <- cbind(bandwidths, local_R2)

write.table(bw_grf,
            file = "Bandwidths and R2 of Local Model.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )


# Q. Optimal mtry selection

opt_mtry <- rf.mtry.optim(
  Au ~ DF.L4.C1 + DF.L4.C2 + DF.L4.C3 + DF.L4.C4 + DF.L4.C5 + DF.L4.C6,
                                  # The local model to be fitted.
  dataset = grf_scale_data,       # Numeric data frame of variables.
  min.mtry = 1,                   # Minimum mtry value for its
                                  # optimization.
  max.mtry = 10,                  # Maximum mtry value for its
                                  # optimization.
  mtry.step = 1,                  # Step in the sequence of mtry
                                  # values for its optimization.
  cv.method = "repeatedcv",       # The resampling method. Options
                                  # are "repeatedcv" and "cv".
  cv.folds = 10                   # Number of folds.
  )


# R. Implement geographical random forest (GRF)

model <- grf(
  Au ~ DF.L4.C1 + DF.L4.C2 + DF.L4.C3 + DF.L4.C4 + DF.L4.C5 + DF.L4.C6,
                                  # The local model to be fitted.
  dframe = grf_scale_data,        # Numeric data frame of variables.
  bw = 23,                        # A positive number that may be an
                                  # integer in the case of an
                                  # "adaptive kernel" or a real in
                                  # the case of a "fixed kernel". In
                                  # the first case, the integer
                                  # denotes the number of nearest
                                  # neighbours, whereas in the
                                  # latter case the real number
                                  # refers to the bandwidth (in
                                  # meters).
  kernel = "adaptive",            # The kernel to be used in the
                                  # regression. Options are
                                  # "adaptive" (default) or "fixed".
  coords = Coords,                # A numeric matrix or data frame
                                  # of two columns giving the X,Y
                                  # coordinates of the observations.
  trees = 10,                     # Number of trees to grow for each
                                  # of the local random forests.
  mtry = NULL,                    # Number of variables randomly
                                  # sampled as candidates at each
                                  # split. The default value is p/3,
                                  # where p is number of variables
                                  # in the formula.
  importance = "impurity",        # Feature importance of the
                                  # dependent variables used as
                                  # input at the random forest.
                                  # Default value is "impurity"
                                  # which refers to the Gini index
                                  # for classification and the
                                  # variance of the responses for
                                  # regression.
  nthreads = 1,                   # Number of threads. Default is
                                  # number of CPUs available.
  forests = FALSE,                # An option to save and export
                                  # (TRUE) or not (FALSE) all the
                                  # local forests.
  weighted = TRUE                 # If TRUE the algorithm calculates
                                  # GRF using the case.weights
                                  # option of the package ranger. If
                                  # FALSE it will calculate local
                                  # random forests without weighting
                                  # each observation in the local
                                  # dataset.
  )

write.table(model[["LGofFit"]],
            file = "Residuals and Local Goodness of Fit.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )

write.table(model[["LocalModelSummary"]],
            file = "Local Model Summary.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = TRUE,
            col.names = TRUE
            )


# S. Calculate spatially aware anomaly scores

res_abs <- abs(model[["LGofFit"]][["LM_ResPred"]])

res_abs <- as.data.frame(res_abs)

anomaly_dan_grf <- cbind(Coords, res_abs)

write.table(anomaly_dan_grf,
            file = "Anomaly by DAN-GRF.txt",
            append = FALSE,
            sep = " ",
            dec = ".",
            row.names = FALSE,
            col.names = TRUE
            )
