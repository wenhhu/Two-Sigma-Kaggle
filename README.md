# Two-Sigma-Kaggle

This repo stores my main script and unorgnized EAD for two sigma kaggle modeling challenge.

## Overview

In competition, the dataset provided in this competition contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id. Time is represented by the 'timestamp' feature and the variable to predict is 'y'. No further information will be provided on the meaning of the features, the transformations that were applied to them, the timescale, or the type of instruments that are included in the data. 

People is supposed to submit his model in terms of python script to kaggle kernel. To avoid any form of forward-looking prediction, the kaggle kernel will evaluate the performance of the model timestamp by timestamp and return the average R. 

## Model Description

In this repo, "Main-script.py" file is the code I submitted. The model I implemented is a two-layer stacking model combining models including robust regression, stepwise linear regression and ExtraTree. Because there are some cross layers in my model structure, I want to describe the model as four layers.

### Layer 0
In the initial layer, we used a robust regression to fit y against golden features that are selected according to mutual information and RF. These models will be included in the layer 1 directly. In addition, we used linear model to stepwisely group features up into covar_sets. The feature grouping is based on the improvement of mse before and after grouping. The grouped features are used to fit y.

### Layer 1
These layer consists of two parts. The first part is the univariate robust regression model. The second part is the extratree's child model fitted from the output of stepwise linear model.

### Layer 2
The top models of layers based on their residual statistics are choosen to be final candidates for forecasting.

### Layer 3
We used an extratree classifier to determine which model in layer 2 to use. The feature we use is the base features and output of stepwise linear model.

## Potential Improvement
After reading the winner's summary, they included market volatility into their model, which is a brilliant synthetic features. I overly constraint my thinking the process and didn't exploited my domain knowledge in finance. 
