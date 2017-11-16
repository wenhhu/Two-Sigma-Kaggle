
# import kagglegym
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics

# Configuration
# ========================================================================================

add_na_indicators = True
add_diff_features = True
na_indicator_cols = ['technical_9', 'technical_0', 'technical_32', 'technical_16',
                     'technical_38', 'technical_44', 'technical_20', 'technical_30', 'technical_13']

diff_cols = ['technical_22', 'technical_20', 'technical_30', 'technical_13',
             'technical_34'] 

univar_rlm_cols = ['technical_22', 'technical_20', 'technical_30_d1', 'technical_20_d1',
                   'technical_30', 'technical_13', 'technical_34']
nr_l2_best_models = 10
wrlm_quant = 0.99
wrlm_min_trainset_fraction = 0.9
wslm_n_group = 30
wslm_max_feat_per_group = 2
wslm_max_abs_y = 0.086
l1_et_n_estimators = 100
l1_et_max_depth = 4
l3_et_n_estimators = 100
l3_et_max_depth = 4
rnd_seed = 17

# Helper functions and objects
# ========================================================================================

# Stepwise regression: Use greedy algorithm to group features up into n_group complementary groups.
# Each group can contain no more than 2 features

class  stepwise_lm:
    def __init__(self, n_group = 30, max_feat_per_group = 2, verbose = True):
        self.n_group = n_group
        self.max_feat_per_group = max_feat_per_group
        self.verbose = verbose

        if self.max_feat_per_group == None:
            self.max_feat_per_group = len(train.columns)
        self.chosen_group = []
        self.models = []

    def fit(self, train, y):
        # We initialize the sets such that each one includes a single random unique feature.

        group_pool = list(train.columns)
        random.shuffle(group_pool)
        for var in group_pool[:self.n_group]:
            self.chosen_group.append([var])
        group_pool = group_pool[self.n_group:]

        # forward stepwise regression:
        # We go over the set of covariates not chosen initially, and see to which set will
        # adding this new covariate improve the training error the most. We then add this
        # new covariate to that set. When the maximum number of covariates
        # max_feat_per_group is reached, that set is no longer considered for adding
        # new variables.

        # initialize mse by set
        best_mses_by_set = np.ones(self.n_group) * float('Inf')

        for var in group_pool:
            gi = 0
            # Declare a list to store the mse of group_sets by adding var
            mses_per_candidate_group = []
            for group_set in self.chosen_group:
                if len(group_set) < self.max_feat_per_group:
                    model = linear_model.LinearRegression(fit_intercept = False,
                                                          normalize = True, copy_X = True, n_jobs = -1)
                    model.fit(train[group_set + [var]], y)
                    mse = metrics.mean_squared_error(y, model.predict(train[group_set +
                                                                            [var]]))
                    mses_per_candidate_group.append(mse)
                else:
                    # If group_set is full, then keep its original mse
                    mses_per_candidate_group.append(best_mses_by_set[gi])
                gi += 1

            gains = best_mses_by_set - mses_per_candidate_group

            if gains.max() > 0:
                temp = gains.argmax()
                # Add var to the group_set which gain the largest mse improvement
                self.chosen_group[temp].append(var)
                best_mses_by_set[temp] = mses_per_candidate_group[temp]

        gsi = 0
        # Fit linear models with all the group_sets
        for group_set in self.chosen_group:
            model = linear_model.LinearRegression(fit_intercept = False, normalize = True,
                                                  copy_X = True, n_jobs = -1)
            model.fit(train[group_set], y)
            self.models.append(model)
            if self.verbose:
                print('Covar set', gsi, 'includes', group_set, 'and achieves',
                      best_mses_by_set[gsi])
            gsi += 1

    def predict(self, data):
        gsi = 0
        for group_set in self.chosen_group:
            data['stacked_lm' + str(gsi)] = self.models[gsi].predict(data[group_set])
            gsi += 1
        # Add predictions of all the linear models with different group_sets to dataframe
        return data

# robust regression, set a threshold size for the train data, if training data is less than
# this threshold or the performance of model is not improved by dropping outliers, we terminate
# the loop

class  robust_lm:
    def __init__(self, quant = 0.999, min_trainset_fraction = 0.9):
        self.quant = quant
        self.min_trainset_fraction = min_trainset_fraction
        self.best_model = []

    def fit(self, train, y):
        tmp_model = linear_model.Ridge(fit_intercept = False)
        best_mse = float('Inf')
        better = True
        train_idxs = train.dropna().index
        min_trainset_fraction = len(train) * self.min_trainset_fraction
        while better:
            tmp_model.fit(train.ix[train_idxs], y.ix[train_idxs])
            mse = metrics.mean_squared_error(tmp_model.predict(train.ix[train_idxs]),
                                             y.ix[train_idxs])
            if mse < best_mse:
                best_mse = mse
                self.best_model = tmp_model
                residuals = y.ix[train_idxs] - tmp_model.predict(train.ix[train_idxs])
                train_idxs = residuals[abs(residuals) <=
                                       abs(residuals).quantile(self.quant)].index
                if len(train_idxs) < min_trainset_fraction:
                    better = False
            else:
                better = False
                self.best_model = tmp_model

    def predict(self, test):
        return self.best_model.predict(test)

# Main routine
# ========================================================================================

print('Initializing')
random.seed(rnd_seed)
env = kagglegym.make()
obs = env.reset()

# Batch supervised training part
# ----------------------------------------------------------------------------------------

with pd.HDFStore("../train.h5", "r") as df:
    train = df.get("train")

# Obtain overall train median per column. We will use this to impute missing values.
train_median = train.median(axis = 0)

print('Adding missing value counts per row')
# Due to imputing, we add a feature to store counts of missing values in each rows to keep this layer of information
train['nr_missing'] = train.isnull().sum(axis = 1)

print('Adding missing value indicators')
# For each column, we add a indicative column to tell whether this column is a missing value
if add_na_indicators:
    for col in na_indicator_cols:
        train[col + '_isna'] = pd.isnull(train[col]).apply(lambda x: 1 if x else 0)
        if len(train[col + '_isna'].unique()) == 1:
            print('Dropped constant missingness indicator:', col, '_isna')
            del train[col + '_isna']
            na_indicator_cols.remove(col)

print('Adding diff features')
if add_diff_features:
    train = train.sort_values(by = ['id', 'timestamp'])
    for col in diff_cols:
        train[col + '_d1'] = train[col].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
    train = train[train.timestamp != 0]

base_features = [x for x in train.columns if x not in ['id', 'timestamp', 'y']]

# Use robust regression to fit y against golden features that are selected according to mutual information and RF
print('Fitting Layer 0 robust univariate linear models')
l0_models = []
l0_columns = []
l0_residuals = []
for col in univar_rlm_cols:
    print('  working on', col)
    model =  robust_lm(quant = wrlm_quant, min_trainset_fraction =
    wrlm_min_trainset_fraction)
    model.fit(train.loc[:, [col]], train.loc[:, 'y'])
    l0_models.append(model)
    l0_columns.append([col])
    l0_residuals.append(abs(model.predict(train[[col]].fillna(train_median)) - train.y))

# Impute all missing values with column median
train = train.fillna(train_median)

# Use stepwise linear model to fit y against groupped noisy features
print('Fitting L0 stepwise linear model')
l0_wslm =  stepwise_lm(n_group = wslm_n_group, max_feat_per_group =
wslm_max_feat_per_group, verbose = True)

# Drop outlying response values from the trainset.
train_idx = train[abs(train.y) < wslm_max_abs_y].index
l0_wslm.fit(train.ix[train_idx, base_features], train.ix[train_idx, 'y'])

l0_wslm_fitted_and_base_features = l0_wslm.predict(train[base_features])
l0_wslm_fitted_and_base_features_cols = l0_wslm_fitted_and_base_features.columns

# Use extratree model to fit y against output from stepwise linear model
print('Training L1 ExtraTree')
model = ensemble.ExtraTreesRegressor(n_estimators = l1_et_n_estimators,
                                     max_depth = l1_et_max_depth, n_jobs = -1, random_state = rnd_seed, verbose = 0)
model.fit(l0_wslm_fitted_and_base_features, train.y)

# The child model of extra tree are taken as part of the first layer model
l1_models = []
l1_columns = []
l1_residuals = []
for extra_tree in model.estimators_:
    l1_models.append(extra_tree)
    l1_columns.append(l0_wslm_fitted_and_base_features_cols)
    l1_residuals.append(abs(extra_tree.predict(l0_wslm_fitted_and_base_features) - train.y))

# Merge univariate robust regression model into layer 1 model
l01_models = l0_models + l1_models
l01_columns = l0_columns + l1_columns
l01_residuals = l0_residuals + l1_residuals

# Model selection layer, which select the optimal model in layer 1 based on their residuals
print('Training L2 select top models')
midxs = np.argmin(np.array(l01_residuals).T, axis = 1)
midxs = pd.Series(midxs).value_counts().head(nr_l2_best_models).index
l2_best_models = []
l2_best_model_columns = []
l2_best_model_residuals = []
for midx in midxs:
    l2_best_models.append(l01_models[midx])
    l2_best_model_columns.append(l01_columns[midx])
    l2_best_model_residuals.append(l01_residuals[midx])

l2_best_model_idx = np.argmin(np.array(l2_best_model_residuals).T, axis = 1)

# Fit a extratree classifier to forecast the optimal model based on robust regression's ouput
print('Training L3 ExtraTree')
l3_et = ensemble.ExtraTreesClassifier(n_estimators = l3_et_n_estimators,
                                      max_depth = l3_et_max_depth, n_jobs = -1, random_state = rnd_seed, verbose = 0)
l3_et.fit(l0_wslm_fitted_and_base_features, l2_best_model_idx)

print('Predicting on holdout set')
oidx = 0
nr_positive_rewards = 0
holdout_rewards = []
prev_diff_cols_data = train[train.timestamp == max(train.timestamp)][['id'] + diff_cols].copy()

# Predicting on holdout dataset timestamp by timestamp
while True:
    oidx += 1
    test = obs.features

    # Preprocess
    test['nr_missing'] = test.isnull().sum(axis = 1)
    if add_na_indicators:
        for elt in na_indicator_cols:
            test[elt + '_isna'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)

    test = test.fillna(train_median)

    if add_diff_features:
        ids_with_prev = list(set(prev_diff_cols_data.id) & set(test.id))
        prev_diff_cols_data = pd.concat([
            test[test.id.isin(ids_with_prev)]['id'],
            pd.DataFrame(
                test[diff_cols][test.id.isin(ids_with_prev)].values -
                prev_diff_cols_data[diff_cols][prev_diff_cols_data.id.isin(ids_with_prev)].values,
                columns = diff_cols, index = test[test.id.isin(ids_with_prev)].index
            )
        ], axis = 1)
        test = test.merge(right = prev_diff_cols_data, how = 'left', on = 'id',
                          suffixes = ('', '_d1')).fillna(0)
        prev_diff_cols_data = test[['id'] + diff_cols].copy()

    # Pass the data through the stacked model to generate a prediction
    l0_wslm_fitted_and_base_features = l0_wslm.predict(test[base_features])
    l3_preds = l3_et.predict_proba(l0_wslm_fitted_and_base_features.loc[:,
                                   l0_wslm_fitted_and_base_features_cols])
    pred = obs.target
    for idx, mdl in enumerate(l2_best_models):
        pred['y'] += (l3_preds[:, idx] *
                      mdl.predict(l0_wslm_fitted_and_base_features[l2_best_model_columns[idx]]))

    obs, reward, done, info = env.step(pred)

    holdout_rewards.append(reward)

    if reward > 0:
        nr_positive_rewards += 1

    if oidx % 100 == 0:
        print('Step', oidx, '#pos', nr_positive_rewards, 'curr', reward, 'mean so far',
              np.mean(holdout_rewards))

    if done:
        print('Done. Public score:', info['public_score'])
        break