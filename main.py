import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut, GridSearchCV,train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV
import matplotlib.font_manager
from sklearn.metrics import r2_score
import os
import operator


data = pd.read_csv("BNSTsigpower.csv")
features = data.columns.tolist()
X = data[features[1:7]]
test_X = data[features[1:7]][0:8]
Y1 = data['HAMA%']
Y2 = data['HAMD%']
Y3 = data['MADRS%']
Y4 = data['DARS%']

test_Y1 = data["HAMA_1y"].dropna(axis = 0, how = "any")
test_Y2 = data["HAMD_1y"].dropna(axis = 0, how = "any")
test_Y3 = data["MADRS_1y"].dropna(axis = 0, how = "any")
test_Y4 = data["DARS_1y"].dropna(axis = 0, how = "any")

df = pd.DataFrame(features, columns = ['fetures_name'])
#feature_names = list(["BNST.θ","BNST.α","BNST.lβ","BNST.hβ","BNST.γ",
                   #   "EEG.θ","EEG.α","EEG.lβ","EEG.hβ","EEG.γ",
                    #  "EEGBNSTCoh.θ","EEGBNSTCoh.α","EEGBNSTCoh.lβ","EEGBNSTCoh.hβ","EEGBNSTCoh.γ"])
feature_names = list(["BNST.θ","BNST.lβ","BNST.hβ","BNST.γ","EEG.θ","EEGBNSTCoh.θ"])

## #######################
#Model = SVR(kernel = "poly",degree = 3)
#Model = RandomForestRegressor(n_estimators=50,min_samples_split = 2, min_samples_leaf = 1, max_depth = None, random_state=42)
#kf = LeaveOneOut()
#kf = KFold(n_splits = 10, shuffle = True)
#score_ndarray = cross_val_score(Model, X, Y, cv = kf, scoring = "explained_variance")
#print(score_ndarray.mean())

def ridgereg(X, Y, testX, testY): #
    lamdas =np.logspace(-5,2,500)
    ridgecv = RidgeCV(alphas = lamdas, cv = 3)
    ridge = ridgecv.fit(X,Y)
    best_ridgealpha = ridge.alpha_
    Model = Ridge(alpha = best_ridgealpha, max_iter = 10000)
    Model.fit(X,Y)
    result = permutation_importance(Model, X, Y,
                                    n_repeats=10, random_state=42, n_jobs=2)
    importance = pd.Series(result.importances_mean, index = feature_names)
    R2score = r2_score(Y,Model.predict(X))
    R2score_test = r2_score(testY, Model.predict(testX))
    return result, importance, R2score, R2score_test

def ranforest(trainx, trainy, testx, testy):
    Model = RandomForestRegressor(n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_depth = None, random_state = 42)
    Model.fit(trainx, trainy)
    result = permutation_importance(Model, trainx, trainy,
                                    n_repeats=10, random_state=42, n_jobs=2)
    importance = pd.Series(result.importances_mean, index = feature_names)
    R2score = r2_score(trainy, Model.predict(trainx))
    R2score_test = r2_score(testy, Model.predict(testx))
    return result, importance, R2score, R2score_test


[result1, importance1, R2score1, R2score_test1] = ridgereg(X, Y1, test_X, test_Y1)
[result2, importance2, R2score2, R2score_test2] = ridgereg(X, Y2, test_X, test_Y2)
[result3, importance3, R2score3, R2score_test3] = ridgereg(X, Y3, test_X, test_Y3)
[result4, importance4, R2score4, R2score_test4] = ridgereg(X, Y4, test_X, test_Y4)
## ########################

plt.rcParams['font.sans-serif'] = 'Helvetica' # 设置全局字体，会被局部字体顶替
font1 = {'family': 'Nimbus Roman',
         'weight': 'bold',
		 'style':'normal',
         'size': 12,
         }

font2 = {'family': 'Helvetica',
         'weight': 'bold',
		 'style':'oblique',
         'size': 10,
         }

fig, ax = plt.subplots(1, 4, dpi=300)
ax[0].set_title("""
HAMA
$\mathregular{R^2}$ score = 0.84
""", font2)
ax[0].spines['top'].set_visible(False)                   # 不显示图表框的上边框
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)                   # 不显示图表框的上边框
ax[0].spines['left'].set_visible(False)
importance1.plot.barh(xerr=result1.importances_std,
                      ax = ax[0], align = "center", color = "#FAD7DD")
ax[0].set_xticks([0, 2])
ax[0].set_yticklabels(feature_names, size = 10)

ax[1].set_title("""
HAMD
$\mathregular{R^2}$ score = 0.96
""", font2)
ax[1].spines['top'].set_visible(False)                   # 不显示图表框的上边框
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)                   # 不显示图表框的上边框
ax[1].spines['left'].set_visible(False)
importance2.plot.barh(xerr=result2.importances_std,
                      ax = ax[1], align = "center", color = "#90D7EB")
ax[1].get_yaxis().set_visible(False)

ax[2].set_title("""
MADRS
$\mathregular{R^2}$ score = 0.50
""", font2)
ax[2].spines['top'].set_visible(False)                   # 不显示图表框的上边框
ax[2].spines['right'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)                   # 不显示图表框的上边框
ax[2].spines['left'].set_visible(False)
importance3.plot.barh(xerr=result3.importances_std,
                      ax = ax[2], align = "center", color = "#B7DBEA")
ax[2].get_yaxis().set_visible(False)


ax[3].set_title("""DARS
$\mathregular{R^2}$ score = 0.09
""", font2)
ax[3].spines['top'].set_visible(False)                   # 不显示图表框的上边框
ax[3].spines['right'].set_visible(False)
ax[3].spines['bottom'].set_visible(False)                   # 不显示图表框的上边框
ax[3].spines['left'].set_visible(False)
importance4.plot.barh(xerr=result4.importances_std,
                      ax = ax[3], align = "center", colormap = "Accent")
ax[3].get_yaxis().set_visible(False)
ax[3].set_xticks([0, 2])
fig.suptitle("Feature importance", x = 0.1, y = -0.5
             , fontdict = font2, ha = "left", va = "bottom"
             ) # 为什么标题没显示出来？
fig.subplots_adjust(wspace = 0.5)
fig.tight_layout()
plt.show()




