import numpy as np
import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt # import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # "pip install scikit-learn", not "pip install sklearn"
from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split 
from sklearn.dummy import DummyClassifier 
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

# файлы отличаются соотношением включенности двух сенсоров, пока берем один файл.
# внутри файла есть набор данных для разных соотношений спирта и воздуха, и указывается какой именно спирт
# то есть надо пять колонок спиртов заместить на одну колонку с названием спирта? Дописать сзади и удалить эти?

alcohols = ["1-Octanol", "1-Propanol", "2-Butanol", "2-propanol", "1-isobutanol"]
#concentrarions = [0.200, 0.300, 0.400, 0.500, 0.600]
#conc_columns = ['0.799_0.201', '0.799_0.201.1', '0.700_0.300', '0.700_0.300.1', '0.600_0.400', '0.600_0.400.1', '0.501_0.499', '0.501_0.499.1', '0.400_0.600', '0.400_0.600.1'] 
concentrations = {'0.799_0.201':0.200, '0.799_0.201.1':0.200, '0.700_0.300':0.300, '0.700_0.300.1':0.300, '0.600_0.400':0.400, '0.600_0.400.1':0.400, '0.501_0.499':0.500, '0.501_0.499.1':0.500, '0.400_0.600':0.600, '0.400_0.600.1':0.600}

# Сенсоры тоже надо перевести в цифру:
# Sensor name MIP ratio NP ratio -> chapter of MIP MIP/NP
# QCM3 1 1 => 0.5
# QCM6 1 0 => 1.0
# QCM7 1 0.5 => 0.67
# QCM10 1 2 => 0.33
# QCM12 0 1 => 0.0

# это надо было в словарь... https://stackoverflow.com/questions/36244380/enumerate-for-dictionary-in-python 
# sensors = [0.5, 1.0, 0.67, 0.33, 0.0]
# files = ['QCM3.csv', 'QCM6.csv', 'QCM7.csv', 'QCM10.csv', 'QCM12.csv']
sensors = {'QCM3': 0.5, 'QCM6': 1.0, 'QCM7': 0.67, 'QCM10': 0.33, 'QCM12': 0.0}

file_size = 250 # unsafe if lenght of files differs

#  берем таблицу (или сразу имя файла?), тип сенсора (а тогда уж и какие-то другие параметры...), и возвращаем в формате "концентрация газа, тип сенсора, тип спирта, показания датчика."
def convert_csv():

    return(0)



def draw_3d(data):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for f in data:
        ax.scatter(f[0], f[1], f[2], c='b')
    plt.show()



# Data load =========================

data_lst = [] # data_lst.append(j)
df_columns = ['Concentration', 'Readings', 'Sensor', '1-Octanol', '1-Propanol', '2-Butanol', '2-propanol', '1-isobutanol', 'Alcohol']

for sens_name in sensors:
    df = pd.read_csv(sens_name + '.csv', sep=';')
    df_format = pd.DataFrame(columns = df_columns, index = [i for i in range(0, 2 * len(df.index))])

#    for i in range(len(concentrations) * len(df.index)):
#        df_format.at[i,"Sensor"] = sensors[sens_name] 
    
    for i in range(df.shape[0]):
        for j in alcohols:
            for k in range(10):
                df_format.at[i*10 + k, j] = df.at[i,j]
                if df.at[i, j] == 1:
                    df_format.at[i*10 + k, "Alcohol"] = j

        for index, (conc_name, conc) in enumerate(concentrations.items()):
            df_format.at[i*10 + index, "Concentration"] = conc
            df_format.at[i*10 + index, "Readings"] = df.at[i, conc_name]
            
    df_format['Concentration'] = pd.to_numeric(df_format['Concentration'], errors='ignore') # Using errors=’coerce’. It will replace all non-numeric values with NaN
    df_format['Sensor'] = pd.to_numeric(df_format['Sensor'], errors='ignore')
    df_format['Readings'] = pd.to_numeric(df_format['Readings'], errors='ignore')
    df_format['1-Octanol'] = pd.to_numeric(df_format['1-Octanol'], errors='ignore')
    df_format['1-Propanol'] = pd.to_numeric(df_format['1-Propanol'], errors='ignore')
    df_format['2-Butanol'] = pd.to_numeric(df_format['2-Butanol'], errors='ignore')
    df_format['2-propanol'] = pd.to_numeric(df_format['2-propanol'], errors='ignore')
    df_format['1-isobutanol'] = pd.to_numeric(df_format['1-isobutanol'], errors='ignore')
   
    tpl = (sens_name, df_format)
   
    data_lst.append(tpl)
    
   
    
#print("data_lst[0] = \n", data_lst[0])
#print("data_lst[3] = \n", data_lst[3])

# !!!!  все что ниже закомментировано - не будет работать тк надо тоже цикл добавить для объектов списка.   !!!!

# <WORKS> 
#df_full.to_csv('df_full2.csv')

# <WORKS> try  Scatterplots with targets =-=-=-==========================

# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
# for idx, feature in enumerate(df_full.columns):
    # df_full.plot(feature, "Readings", subplots=True, kind="scatter", ax=axes[idx // 4, idx % 4])
# plt.show()




# <WORKS> Correlation matrix  =-=-=-==========================

# plt.figure(figsize=(10,10))
# sns.heatmap(num_cols.corr(), cmap="RdYlBu_r");
# plt.show()


#  <WORKS> PCA(n_components=3) ===============================

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(num_cols)

# model = PCA(n_components=3)
# model.fit(df_scaled)
# df_pca_3 = model.transform(df_scaled)

# draw_3d(df_pca_3)


# <WORKS!> Class visualizations with PCA ===============

# df_class = df_full['Sensor'] # Concentration, , Sensor не уверен что имено надо указать , над пробовать)

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(num_cols)

# model = PCA(n_components=3)
# model.fit(df_scaled)
# pca_coords = model.transform(df_scaled)

# df_to_draw = pd.DataFrame({
    # 'class': df_class,
    # 'pca1': pca_coords[:, 0],
    # 'pca2': pca_coords[:, 1],
    # 'pca3': pca_coords[:, 2],    
# })
# colors = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'gold', 'darkorange', 'lime']


# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111, projection='3d')

# for f in df_to_draw.iterrows():
    # ax.scatter(f[1]['pca1'], f[1]['pca2'], f[1]['pca3'], c=colors[int(f[1]['class'])*4]) # for Readings - c=colors[int(f[1]['class'])//50]) is better, for another - smth different from colors[int(f[1]['class'])]
    # # c=colors[int(f[1]['class'])*4] for Sensor, 
# plt.show()


# Learning ========================
'''


x = np.zeros(len(sensors))
y_dummy = np.zeros(len(sensors))
y_log = np.zeros(len(sensors))
#comparison = np.zeros(shape=(len(sensors), 3))

for i, sensor in enumerate(data_lst):
    df = sensor[1]
    sens_name = sensor[0]
    y =  list(df["Alcohol"])
    X = df.iloc[:, :2]

    # Increase the number of iterations (max_iter) or scale the data as shown in:
    #    https://scikit-learn.org/stable/modules/preprocessing.html
    min_max_scaler = preprocessing.MinMaxScaler() 
    X = min_max_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25) # Для воспроизводимости вы должны установить аргумент random_state.
    
    X_train_minmax = min_max_scaler.fit_transform(X_train)  # а это зачем? дальше переменная X_train_minmax не используется...
    
    dummy_clf = DummyClassifier(strategy="most_frequent") 
    dummy_clf.fit(X_train, y_train);
    
    x[i] = sensors[sens_name]
    
    y_pred_dummy = dummy_clf.predict(X_test) 
    acc = accuracy_score(y_test, y_pred_dummy)
    print(sens_name, "accuracy of dummy clf is ", acc)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train); # for np array - y_train.ravel()
    y_dummy[i] = acc

    
    y_pred_log_reg = log_reg.predict(X_test) 
    acc = accuracy_score(y_test, y_pred_log_reg)
    print(sens_name, "accuracy of log reg is ", acc)
    y_log[i] = acc


#comparison[comparison[:, 0].argsort()]


plt.plot(x,y_dummy, 'ko' , label = 'Dummy') 
plt.plot(x,y_log, 'r*' , label = 'Log reg') 
plt.legend(loc = 'center right')

plt.xlabel('Sensor')
plt.ylabel('Accuracy')

plt.show()
'''


'''
QCM3 accuracy of dummy clf is  0.12698412698412698
QCM3 accuracy of log reg is  0.42857142857142855
QCM6 accuracy of dummy clf is  0.12698412698412698
QCM6 accuracy of log reg is  0.4444444444444444
QCM7 accuracy of dummy clf is  0.12698412698412698
QCM7 accuracy of log reg is  0.38095238095238093
QCM10 accuracy of dummy clf is  0.12698412698412698
QCM10 accuracy of log reg is  0.4444444444444444
QCM12 accuracy of dummy clf is  0.12698412698412698
QCM12 accuracy of log reg is  0.3968253968253968
'''

# Optimising ========================

from sklearn.metrics import classification_report


# Grid Search

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV




from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


precision = []


pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)), # increased from default (100) https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
    ]
)


for i, sensor in enumerate(data_lst):
    df = sensor[1]
    sens_name = sensor[0]
        
#    if i > 1:
#        break
        
    y =  list(df["Alcohol"])
    X = df.iloc[:, :2]
    
    # Increase the number of iterations (max_iter) or scale the data as shown in:
    #    https://scikit-learn.org/stable/modules/preprocessing.html
    min_max_scaler = preprocessing.MinMaxScaler() 
    X = min_max_scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25) # Для воспроизводимости вы должны установить аргумент random_state.
 #   X_train_minmax = min_max_scaler.fit_transform(X_train)  
 
    model = pipeline.fit(X_train, y_train)
    y_pred_simple = model.predict(X_test)
    print(classification_report(y_test, y_pred_simple))
   
    '''
                  precision    recall  f1-score   support
    
       1-Octanol       1.00      1.00      1.00        13
      1-Propanol       0.29      0.62      0.40         8
    1-isobutanol       0.00      0.00      0.00        14
       2-Butanol       0.60      0.86      0.71        14
      2-propanol       0.92      0.79      0.85        14
    
        accuracy                           0.65        63
       macro avg       0.56      0.65      0.59        63
    weighted avg       0.58      0.65      0.60        63
    '''

# Оптимизация гиперпараметров

    parameters = {
        'scaler__with_mean': [True, False],
        'clf__C': np.linspace(0.01, 1, 10),
        'clf__penalty': ['l2'], #  было 'clf__penalty': ['l2', None] - надоели предупреждения "Setting penalty=None will ignore the C and l1_ratio parameters" 
        'clf__random_state': [2023],
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        n_jobs=2,
        verbose=1,
    )
    
#    print(%%time) # prints the wall time for the entire cell whereas %time gives you the time for first line only
    
    print('grid_search:\n', grid_search.fit(X_train, y_train))
    

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("param_name and best_parameters[param_name]:\n")
        print(f"{param_name}: {best_parameters[param_name]}")
    
    y_pred_optimized = grid_search.best_estimator_.predict(X_test)
    print('grid classification_report: \n', classification_report(y_test, y_pred_optimized))
    '''
    grid_search:
     GridSearchCV(estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('clf', LogisticRegression())]),
                 n_jobs=2,
                 param_grid={'clf__C': array([0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.  ]),
                             'clf__penalty': ['l2', 'none'],
                             'clf__random_state': [2023],
                             'scaler__with_mean': [True, False]},
                 verbose=1)
    
    param_name and best_parameters[param_name]:
    
    clf__C: 0.01
    param_name and best_parameters[param_name]:
    
    clf__penalty: none
    param_name and best_parameters[param_name]:
    
    clf__random_state: 2023
    param_name and best_parameters[param_name]:
    
    scaler__with_mean: True
    
    
    classification_report (100 iterations): 
                   precision    recall  f1-score   support
    
       1-Octanol       1.00      1.00      1.00        13
      1-Propanol       0.29      0.62      0.40         8
    1-isobutanol       0.00      0.00      0.00        14
       2-Butanol       0.60      0.86      0.71        14
      2-propanol       0.92      0.79      0.85        14
    
        accuracy                           0.65        63
       macro avg       0.56      0.65      0.59        63
    weighted avg       0.58      0.65      0.60        63


    grid classification_report (1000 and 3000 iterations): 
                   precision    recall  f1-score   support
    
       1-Octanol       1.00      1.00      1.00        13
      1-Propanol       0.45      0.62      0.53         8
    1-isobutanol       0.50      0.21      0.30        14
       2-Butanol       0.72      0.93      0.81        14
      2-propanol       0.80      0.86      0.83        14
    
        accuracy                           0.73        63
       macro avg       0.70      0.72      0.69        63
    weighted avg       0.71      0.73      0.70        63



    '''
    

# Randomized Search



    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=10,
        random_state=2023,
        n_jobs=4,
        verbose=1,
    )

    random_search.fit(X_train, y_train)

    
    best_parameters = random_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print(f"{param_name}: {best_parameters[param_name]}")
        
    y_pred_optimized = grid_search.best_estimator_.predict(X_test)
    report = classification_report(y_test, y_pred_optimized)
    print('random classification_report: \n', report)
 
   
    '''
    random classification_report: 
               precision    recall  f1-score   support

   1-Octanol       1.00      1.00      1.00        13
  1-Propanol       0.29      0.62      0.40         8
1-isobutanol       0.00      0.00      0.00        14
   2-Butanol       0.60      0.86      0.71        14
  2-propanol       0.92      0.79      0.85        14

    accuracy                           0.65        63
   macro avg       0.56      0.65      0.59        63
weighted avg       0.58      0.65      0.60        63
    '''
    
    report_dict = classification_report(y_test, y_pred_optimized, output_dict=True)
    print('random classification_report dict: \n', report_dict)
    '''
random classification_report dict: 
 {'1-Octanol': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 13}, '1-Propanol': {'precision': 0.29411764705882354, 'recall': 0.625, 'f1-score': 0.4, 'support': 8}, '1-isobutanol': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 14}, '2-Butanol': {'precision': 0.6, 'recall': 0.8571428571428571, 'f1-score': 0.7058823529411764, 'support': 14}, '2-propanol': {'precision': 0.9166666666666666, 'recall': 0.7857142857142857, 'f1-score': 0.8461538461538461, 'support': 14}, 'accuracy': 0.6507936507936508, 'macro avg': {'precision': 0.562156862745098, 'recall': 0.6535714285714286, 'f1-score': 0.5904072398190044, 'support': 63}, 'weighted avg': {'precision': 0.5807345160286337, 'recall': 0.6507936507936508, 'f1-score': 0.6020397902750844, 'support': 63}}
    '''    
     
     # удалить из словаря:
#         'accuracy': 0.6507936507936508, 
     # 'macro avg': {'precision': 0.562156862745098, 'recall': 0.6535714285714286, 'f1-score': 0.5904072398190044, 'support': 63}, 
#     'weighted avg': {'precision': 0.5807345160286337, 'recall': 0.6507936507936508, 'f1-score': 0.6020397902750844, 'support': 63}
    
    report_dict.pop('accuracy') 
    report_dict.pop('macro avg')
    report_dict.pop('weighted avg')
    precision_tpl = (sens_name, report_dict)
    precision.append(precision_tpl)
     
'''    
# Halving Grid Search¶

  #  X, y = make_classification(n_samples=400, n_features=12, random_state=0)
    
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    
    param_dist = {
        "max_depth": [3, None],
        "max_features": np.arange(1, 6),
        "min_samples_split": np.arange(2, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }
    
    rsh = HalvingRandomSearchCV(
        estimator=clf,
        param_distributions=param_dist, 
        factor=2, 
        random_state=0,
    )
    rsh.fit(X, y)
    

    
    results = pd.DataFrame(rsh.cv_results_)
    
    print('Halving Grid Search report: \n', results)
        
    results["params_str"] = results.params.apply(str)
    results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
    mean_scores = results.pivot(
        index="iter", 
        columns="params_str",
         values="mean_test_score",
    )
    ax = mean_scores.plot(legend=False, alpha=0.6)
    
    labels = [
        f"iter={i}\nn_samples={rsh.n_resources_[i]}\nn_candidates={rsh.n_candidates_[i]}"
        for i in range(rsh.n_iterations_)
    ]
    
    ax.set_xticks(range(rsh.n_iterations_))
    ax.set_xticklabels(labels, rotation=45, multialignment="left")
    ax.set_title("Scores of candidates over iterations")
    ax.set_ylabel("Mean test score", fontsize=15)
    ax.set_xlabel("Iterations", fontsize=15)
    plt.legend(loc = 'center right')
    plt.tight_layout()
    plt.grid()
    plt.show()

'''
# в итоге какая-то ерунда. надо в таблицу все экспортировать и построить график, сравнивающий эффективность по разным сенсорам и спиртам
# https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format 
# взять из каждого такого словаря параметр точности для каждого спирта и построить график для сенсоров: {'1-Octanol': {'precision': 1.0,
# перебор по списку спиртов, обращаясь к каждому элементу словаря и внося этот параметр в .... будет 6 таблиц по числу сенсоров или спиртов? Или спирты - цвет значков? 
# и  там же нарисовать что-то среднее по распознаванию, Какой сенсор с вреднем лучше для всех спиртов


'''
As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply having the report returned as a dict:

report = classification_report(y_test, y_pred, output_dict=True)

and then construct a Dataframe and transpose it:

df = pandas.DataFrame(report).transpose()

From here on, you are free to use the standard pandas methods to generate your desired output formats (CSV, HTML, LaTeX, ...).

See the documentation.
'''
#print(precision)


# prediction:
    


# x у нас это спирты или сенсоры? Пусть будут сенсоры
# sensor_scale = [] # добавлять в конце цикла туда название, то что sens_name = sensor[0]
# или создавать снова словарь? precision = {сенсор, [{спирт: точность}, ...]}
#  {'1-Octanol': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 13}, '1-Propanol': {'precision': 0.29411764705882354, 'recall': 0.625, 'f1-score': 0.4, 'support': 8}, '1-isobutanol': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 14}, '2-Butanol': {'precision': 0.6, 'recall': 0.8571428571428571, 'f1-score': 0.7058823529411764, 'support': 14}, '2-propanol': {'precision': 0.9166666666666666, 'recall': 0.7857142857142857, 'f1-score': 0.8461538461538461, 'support': 14}, 'accuracy': 0.6507936507936508, 'macro avg': {'precision': 0.562156862745098, 'recall': 0.6535714285714286, 'f1-score': 0.5904072398190044, 'support': 63}, 'weighted avg': {'precision': 0.5807345160286337, 'recall': 0.6507936507936508, 'f1-score': 0.6020397902750844, 'support': 63}}

#    precision_tpl = (sens_name, report_dict)
#    precision.append(precision_tpl)

#plt.plot(precision[0],y_dummy, 'ko' , label = 'Dummy') 

#for sens in precision:
#    print('sens[0]: \n', sens[0])
#    print('sens[1]: \n', sens[1])    
#    for alc, param in sens[1].items():  
#        print('alc, param: \n', alc, param)
#        for par, val in param.items():
##            if val == 'precision':
#            print('par, val: \n', par, val)

color_lst = ['b*', 'g*', 'r*', 'm*', 'k*']    
for sens in precision:
    for i, (alc, param) in enumerate(sens[1].items()):
        plt.plot(sens[0], param['precision'], color_lst[i])
        
        

        
# явно надо спирты в отдельный список сгруппировать...        

# plt.plot(precision[0],y_log, 'r*' , label = 'Log reg') 
plt.legend(alcohols, loc = 'center right')

plt.xlabel('Sensor')
plt.ylabel('Accuracy')

plt.show()



