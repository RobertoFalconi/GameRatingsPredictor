# -*- coding: utf-8 -*-

import itertools
import threading
import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

RUN_ANALYSIS_LOGISTIC_PENALTY = False
RUN_ANALYSIS_RANDOMFOREST_NUMBERESTIMATOR = False
RUN_ANALYSIS_RANDOMFOREST_ENTROPY = False
RUN_ANALYSIS_RANDOMFOREST_ENTROPYANDNE40 = False
RUN_ANALYSIS_KNN_NUMBERNEIGHBORS = False
RUN_ANALYSIS_KNN_LEAFSIZE = False
RUN_ANALYSIS_KNN_P = False

RUN_CROSSVALIDATION = False

PRINT_DATASET_ANALYSIS = False
PRINT_LEARNINGCURVES = False


def animate(i, stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        sys.stdout.write('\r' + c + ' Sistemo il dataset... ' + c)
        sys.stdout.flush()
        time.sleep(0.1)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def main():
    print('\nAvvio del programma\n')

    t_stop = threading.Event()
    t = threading.Thread(target=animate, args=(1, t_stop))
    t.start()

    # Caricamento del dataset
    df = pd.read_csv('../GameRatingsPredictor/docs/Video_Games_Sales_as_at_22_Dec_2016.csv', encoding="utf-8")

    # Dataset analysis

    if PRINT_DATASET_ANALYSIS:
        # https://www.kaggle.com/etakla/d/rush4ratio/video-game-sales-with-ratings/exploring-the-dataset-bivariate-analysis

        df = df[df.Rating != "RP"]
        df = df[df.Rating != "EC"]
        df = df[df.Rating != "AO"]
        df = df[df.Rating != "K-A"]

        ax = df.groupby('Rating').sum().unstack().Global_Sales.sort_values(ascending=False).head(10).plot(kind='bar',
                                                                                                          figsize=(
                                                                                                              13, 5));
        for p in ax.patches:
            ax.annotate(
                str(round(p.get_height())) + "\n" + str(round(100.0 * p.get_height() / df.NA_Sales.sum())) + "%",
                (p.get_x() + 0.13, p.get_height() - 85),
                color='black', fontsize=12, fontweight='bold')

        rating_sales_percentages_by_year = (df.groupby(['Year_of_Release', 'Rating']).Global_Sales.sum()) * \
                                           (100) / df.groupby(['Year_of_Release']).Global_Sales.sum()
        rating_sales_percentages_by_year.unstack().plot(kind='area', stacked=True, colormap='Spectral',
                                                        figsize=(13, 4));

        # temp is the sum of all variables for each rating by year
        temp = df.groupby(['Year_of_Release', 'Rating']).sum().reset_index().groupby('Year_of_Release')

        platform_yearly_winner_df = pd.DataFrame()

        for year, group in temp:
            current_year = temp.get_group(year)
            this_year_max_sales = 0.0
            current_year_winner = ""
            row = {'year': "", 'winner': "", 'sales': ""}
            for index, platform_data in current_year.iterrows():
                if platform_data.Global_Sales > this_year_max_sales:
                    this_year_max_sales = platform_data.Global_Sales
                    current_year_winner = platform_data.Rating

            row['year'] = year
            row['winner'] = current_year_winner
            row['sales'] = this_year_max_sales
            platform_yearly_winner_df = platform_yearly_winner_df.append(row, ignore_index=True)

        plt.figure(figsize=(13, 4))

        g = sns.pointplot(x=platform_yearly_winner_df.year,
                          y=platform_yearly_winner_df.sales,
                          hue=platform_yearly_winner_df.winner);

        # http://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
        g.set_xticklabels(g.get_xticklabels(), rotation=90);

    # Eliminazione dei Platform obsoleti

    df = df[df.Platform != "2600"]
    df = df[df.Platform != "3DO"]
    df = df[df.Platform != "DC"]
    df = df[df.Platform != "GEN"]
    df = df[df.Platform != "GG"]
    df = df[df.Platform != "NG"]
    df = df[df.Platform != "PCFX"]
    df = df[df.Platform != "SAT"]
    df = df[df.Platform != "SCD"]
    df = df[df.Platform != "TG16"]
    df = df[df.Platform != "WS"]

    # Eliminazione elementi non completi

    df = df[df.User_Score != "tbd"]
    df = df.dropna().reset_index(drop=True)

    # Discretizzazione feature Platform

    df["Platform_XB"] = 0
    df["Platform_X360"] = 0
    df["Platform_XOne"] = 0
    df["Platform_PC"] = 0
    df["Platform_PS"] = 0
    df["Platform_PS2"] = 0
    df["Platform_PS3"] = 0
    df["Platform_PS4"] = 0
    df["Platform_PSP"] = 0
    df["Platform_PSV"] = 0
    df["Platform_GB"] = 0
    df["Platform_GBA"] = 0
    df["Platform_DS"] = 0
    df["Platform_3DS"] = 0
    df["Platform_NES"] = 0
    df["Platform_SNES"] = 0
    df["Platform_N64"] = 0
    df["Platform_GC"] = 0
    df["Platform_Wii"] = 0
    df["Platform_WiiU"] = 0

    for elem in df.index.get_values():
        if df.get_value(elem, "Platform") == "XB": df.set_value(elem, "Platform_XB", 1)
        if df.get_value(elem, "Platform") == "X360": df.set_value(elem, "Platform_X360", 1)
        if df.get_value(elem, "Platform") == "XOne": df.set_value(elem, "Platform_XOne", 1)
        if df.get_value(elem, "Platform") == "PS": df.set_value(elem, "Platform_PS", 1)
        if df.get_value(elem, "Platform") == "PS2": df.set_value(elem, "Platform_PS2", 1)
        if df.get_value(elem, "Platform") == "PS3": df.set_value(elem, "Platform_PS3", 1)
        if df.get_value(elem, "Platform") == "PS4": df.set_value(elem, "Platform_PS4", 1)
        if df.get_value(elem, "Platform") == "PSP": df.set_value(elem, "Platform_PSP", 1)
        if df.get_value(elem, "Platform") == "PSV": df.set_value(elem, "Platform_PSV", 1)
        if df.get_value(elem, "Platform") == "GB": df.set_value(elem, "Platform_GB", 1)
        if df.get_value(elem, "Platform") == "GBA": df.set_value(elem, "Platform_GBA", 1)
        if df.get_value(elem, "Platform") == "DS": df.set_value(elem, "Platform_DS", 1)
        if df.get_value(elem, "Platform") == "3DS": df.set_value(elem, "Platform_3DS", 1)
        if df.get_value(elem, "Platform") == "NES": df.set_value(elem, "Platform_NES", 1)
        if df.get_value(elem, "Platform") == "SNES": df.set_value(elem, "Platform_SNES", 1)
        if df.get_value(elem, "Platform") == "N64": df.set_value(elem, "Platform_N64", 1)
        if df.get_value(elem, "Platform") == "GC": df.set_value(elem, "Platform_GC", 1)
        if df.get_value(elem, "Platform") == "Wii": df.set_value(elem, "Platform_Wii", 1)
        if df.get_value(elem, "Platform") == "WiiU": df.set_value(elem, "Platform_WiiU", 1)

    # Discretizzazione feature Genre

    df["Genre_Action"] = 0
    df["Genre_Adventure"] = 0
    df["Genre_Fighting"] = 0
    df["Genre_Misc"] = 0
    df["Genre_Platform"] = 0
    df["Genre_Puzzle"] = 0
    df["Genre_Shooter"] = 0
    df["Genre_Sports"] = 0
    df["Genre_Simulation"] = 0
    df["Genre_Strategy"] = 0
    df["Genre_Racing"] = 0
    df["Genre_Role-Playing"] = 0

    for elem in df.index.get_values():
        if df.get_value(elem, "Genre") == "Action": df.set_value(elem, "Genre_Action", 1)
        if df.get_value(elem, "Genre") == "Adventure": df.set_value(elem, "Genre_Adventure", 1)
        if df.get_value(elem, "Genre") == "Fighting": df.set_value(elem, "Genre_Fighting", 1)
        if df.get_value(elem, "Genre") == "Misc": df.set_value(elem, "Genre_Misc", 1)
        if df.get_value(elem, "Genre") == "Platform": df.set_value(elem, "Genre_Platform", 1)
        if df.get_value(elem, "Genre") == "Puzzle": df.set_value(elem, "Genre_Puzzle", 1)
        if df.get_value(elem, "Genre") == "Shooter": df.set_value(elem, "Genre_Shooter", 1)
        if df.get_value(elem, "Genre") == "Sports": df.set_value(elem, "Genre_Sports", 1)
        if df.get_value(elem, "Genre") == "Simulation": df.set_value(elem, "Genre_Simulation", 1)
        if df.get_value(elem, "Genre") == "Strategy": df.set_value(elem, "Genre_Strategy", 1)
        if df.get_value(elem, "Genre") == "Racing": df.set_value(elem, "Genre_Racing", 1)
        if df.get_value(elem, "Genre") == "Role-Playing": df.set_value(elem, "Genre_Role-Playing", 1)

    # Discretizzazione feature Rating

    df["Rating_Everyone"] = 0
    df["Rating_Everyone10"] = 0
    df["Rating_Teen"] = 0
    df["Rating_Mature"] = 0
    df["Rating_Adult"] = 0

    for elem in df.index.get_values():
        if df.get_value(elem, "Rating") == "E": df.set_value(elem, "Rating_Everyone", 1)
        if df.get_value(elem, "Rating") == "E10+": df.set_value(elem, "Rating_Everyone10", 1)
        if df.get_value(elem, "Rating") == "T": df.set_value(elem, "Rating_Teen", 1)
        if df.get_value(elem, "Rating") == "M": df.set_value(elem, "Rating_Mature", 1)
        if df.get_value(elem, "Rating") == "AO": df.set_value(elem, "Rating_Adult", 1)

    # Discretizzazione feature Publisher

    publisher_list = []

    for elem in df.Publisher:
        if elem not in publisher_list:
            publisher_list.append(elem)

    for elem in publisher_list:
        df[elem] = 0

    for elem in df.index.get_values():
        df.set_value(elem, df.get_value(elem, "Publisher"), 1)

    # Discretizzazione feature Developer

    developer_list = []

    for elem in df.Developer:
        if elem not in developer_list:
            developer_list.append(elem)

    for elem in developer_list:
        df[elem] = 0

    for elem in df.index.get_values():
        df.set_value(elem, df.get_value(elem, "Developer"), 1)

    # Eliminazione elementi inutili

    df = df[df.Rating != 'AO']
    df = df[df.Rating != 'K-A']
    df = df[df.Rating != 'RP']
    df = df[df.Rating != 'EC']

    # Eliminazione feature inutili

    del df['User_Score']
    del df['User_Count']
    del df['Critic_Score']
    del df['Critic_Count']

    # Eliminazione feature discretizzate

    del df['Platform']
    del df['Genre']
    del df['Publisher']
    del df['Developer']
    del df['Name']

    # Eliminazione feature con elementi inferiori a 2

    del df['Rating_Adult']

    # Mischio in maniera casuale il dataset

    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)

    # Salvo le y

    y_true = np.array(df['Rating'])

    # Cambio le classi da E|E10+|T|M in 1|2|3|4 rispettivamente per evitare possibili problemi con
    # la classe E10+ e i caratteri speciali

    y_true_int = np.empty(len(y_true), dtype=int)
    i = 0
    for elem in y_true:
        if elem == "E":
            y_true_int[i] = 1
        elif elem == "E10+":
            y_true_int[i] = 2
        elif elem == "T":
            y_true_int[i] = 3
        else:
            y_true_int[i] = 4
        i += 1

    # Elimino la colonna delle y dal dataset

    del df['Rating']

    # Conversione di tutti i valori degli elementi in float

    df = df.astype('float64')

    # Blocco il thread dell'animazione del testo "Sistemo il dataset"

    t_stop.set()

    sys.stdout.write('\rFatto!                   ')
    time.sleep(1.5)

    # Stampo una breve analisi del dataset

    print("\n\n")
    print("╔══════════════════╗")
    print("║ Dataset Analysis ║")
    print("╚══════════════════╝\n")

    print_e = df["Rating_Everyone"].value_counts()[1]
    print_e10 = df["Rating_Everyone10"].value_counts()[1]
    print_teen = df["Rating_Teen"].value_counts()[1]
    print_mature = df["Rating_Mature"].value_counts()[1]
    # print(df["Rating_Adult"].value_counts())

    print("Number of elements: " + str(print_e + print_e10 + print_teen + print_mature) + "\n")

    df_stampa = pd.DataFrame({"Rating": ['Everyone', 'Everyone 10+', 'Teen', 'Mature'],
                              "Counts": [print_e, print_e10, print_teen, print_mature]})

    cols = df_stampa.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_stampa = df_stampa[cols]

    print(df_stampa)

    #
    #
    #

    # Inizio la computazione sulle classi binarie Rating_Everyone, Rating_Everyone10, Rating_Teen e
    # Rating_Mature da parte di Logistic Regression, Random Forest e k-NN con i parametri impostati
    # come descritto nella relazione

    print("\n")
    print("╔══════════════════════╗")
    print("║ Binary rating values ║")
    print("╚══════════════════════╝")

    print("┌─────────────────┐")
    print("│ RATING EVERYONE │")
    print("└─────────────────┘")

    df2 = df
    y = df2['Rating_Everyone'].values
    df2 = df2.drop(['Rating_Everyone'], axis=1)
    df2 = df2.drop(['Rating_Everyone10'], axis=1)
    df2 = df2.drop(['Rating_Teen'], axis=1)
    df2 = df2.drop(['Rating_Mature'], axis=1)
    X = df2.values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

    if PRINT_LEARNINGCURVES:
        print("Random Forest: ")
        plot_learning_curve(RandomForestClassifier(), "Rating_Everyone Learning Curves (Random Forest)", X, y,
                            n_jobs=-1)
        print("k-NN: ")
        plot_learning_curve(KNeighborsClassifier(), "Rating_Everyone Learning Curves (k-NN)", X, y, n_jobs=-1)

    log_reg1 = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1,
                                  class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    log_reg1.fit(Xtrain, ytrain)
    y_val_l = log_reg1.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)

    print("Logistic Regression Rating_Everyone accuracy: ", ris)
    print("Logistic Regression Rating_Everyone misclassification: ", ytest.size - mis)

    radm1 = RandomForestClassifier(n_estimators=240, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_split=1e-07, bootstrap=True,
                                   oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                                   class_weight=None)
    radm1.fit(Xtrain, ytrain)
    y_val_l = radm1.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Random Forest Rating_Everyone accuracy: ", ris)
    print("Random Forest Rating_Everyone misclassification: ", ytest.size - mis)

    knn1 = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                metric='minkowski', metric_params=None, n_jobs=1)
    knn1.fit(Xtrain, ytrain)
    y_val_l = knn1.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
    print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
    print("\n")

    if RUN_CROSSVALIDATION:

        for clf, label in zip([log_reg1, radm1, knn1], ['Logistic Regression', 'Random Forest', 'k-NN']):
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            print("Cross-validation: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    print()

    #
    #
    #

    print("┌─────────────────────┐")
    print("│ RATING EVERYONE 10+ │")
    print("└─────────────────────┘")

    df2 = df
    y = df2['Rating_Everyone10'].values
    df2 = df2.drop(['Rating_Everyone'], axis=1)
    df2 = df2.drop(['Rating_Everyone10'], axis=1)
    df2 = df2.drop(['Rating_Teen'], axis=1)
    df2 = df2.drop(['Rating_Mature'], axis=1)
    X = df2.values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

    if PRINT_LEARNINGCURVES:
        print("Random Forest: ")
        plot_learning_curve(RandomForestClassifier(), "Rating_Everyone10+ Learning Curves (Random Forest)", X, y,
                            n_jobs=-1)
        print("k-NN: ")
        plot_learning_curve(KNeighborsClassifier(), "Rating_Everyone10+ Learning Curves (k-NN)", X, y, n_jobs=-1)

    log_reg2 = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1,
                                  class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    log_reg2.fit(Xtrain, ytrain)
    y_val_l = log_reg2.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Logistic Regression Rating_Everyone10+ accuracy: ", ris)
    print("Logistic Regression Rating_Everyone10+ misclassification: ", ytest.size - mis)

    radm2 = RandomForestClassifier(n_estimators=240, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_split=1e-07, bootstrap=True,
                                   oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                                   class_weight=None)
    radm2.fit(Xtrain, ytrain)
    y_val_l = radm2.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Random Forest Rating_Everyone10+ accuracy: ", ris)
    print("Random Forest Rating_Everyone10+ misclassification: ", ytest.size - mis)

    knn2 = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                metric='minkowski', metric_params=None, n_jobs=1)
    knn2.fit(Xtrain, ytrain)
    y_val_l = knn2.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
    print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
    print("\n")

    if RUN_CROSSVALIDATION:

        for clf, label in zip([log_reg2, radm2, knn2],['Logistic Regression', 'Random Forest', 'k-NN']):
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            print("Cross-validation: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    print()

    #
    #
    #

    print("┌─────────────┐")
    print("│ RATING TEEN │")
    print("└─────────────┘")

    df2 = df
    y = df2['Rating_Teen'].values
    df2 = df2.drop(['Rating_Everyone'], axis=1)
    df2 = df2.drop(['Rating_Everyone10'], axis=1)
    df2 = df2.drop(['Rating_Teen'], axis=1)
    df2 = df2.drop(['Rating_Mature'], axis=1)
    X = df2.values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

    if PRINT_LEARNINGCURVES:
        print("Random Forest: ")
        plot_learning_curve(RandomForestClassifier(), "Rating_Teen Learning Curves (Random Forest)", X, y, n_jobs=-1)
        print("k-NN: ")
        plot_learning_curve(KNeighborsClassifier(), "Rating_Teen Learning Curves (k-NN)", X, y, n_jobs=-1)

    log_reg3 = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1,
                                  class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    log_reg3.fit(Xtrain, ytrain)
    y_val_l = log_reg3.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Logistic Regression Rating_Teen accuracy: ", ris)
    print("Logistic Regression Rating_Teen misclassification: ", ytest.size - mis)

    radm3 = RandomForestClassifier(n_estimators=240, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_split=1e-07, bootstrap=True,
                                   oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                                   class_weight=None)
    radm3.fit(Xtrain, ytrain)
    y_val_l = radm3.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Random Forest Rating_Teen accuracy: ", ris)
    print("Random Forest Rating_Teen misclassification: ", ytest.size - mis)

    knn3 = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                metric='minkowski', metric_params=None, n_jobs=1)
    knn3.fit(Xtrain, ytrain)
    y_val_l = knn3.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
    print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
    print("\n")

    if RUN_CROSSVALIDATION:

        for clf, label in zip([log_reg3, radm3, knn3],
                              ['Logistic Regression', 'Random Forest', 'k-NN']):
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            print("Cross-validation: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    print()

    #
    #
    #

    print("┌───────────────┐")
    print("│ RATING MATURE │")
    print("└───────────────┘")

    df2 = df
    y = df2['Rating_Mature'].values
    df2 = df2.drop(['Rating_Everyone'], axis=1)
    df2 = df2.drop(['Rating_Everyone10'], axis=1)
    df2 = df2.drop(['Rating_Teen'], axis=1)
    df2 = df2.drop(['Rating_Mature'], axis=1)
    X = df2.values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

    if PRINT_LEARNINGCURVES:
        print("Random Forest: ")
        plot_learning_curve(RandomForestClassifier(), "Rating_Mature Learning Curves (Random Forest)", X, y, n_jobs=-1)
        print("k-NN: ")
        plot_learning_curve(KNeighborsClassifier(), "Rating_Mature Learning Curves (k-NN)", X, y, n_jobs=-1)

    log_reg4 = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=True, intercept_scaling=1,
                                  class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    log_reg4.fit(Xtrain, ytrain)
    y_val_l = log_reg4.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Logistic Regression Rating_Mature accuracy: ", ris)
    print("Logistic Regression Rating_Mature misclassification: ", ytest.size - mis)

    radm4 = RandomForestClassifier(n_estimators=240, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                   min_impurity_split=1e-07, bootstrap=True,
                                   oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False,
                                   class_weight=None)
    radm4.fit(Xtrain, ytrain)
    y_val_l = radm4.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("Random Forest Rating_Mature accuracy: ", ris)
    print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

    knn4 = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                metric='minkowski', metric_params=None, n_jobs=1)
    knn4.fit(Xtrain, ytrain)
    y_val_l = knn4.predict(Xtest)
    ris = accuracy_score(ytest, y_val_l)
    mis = accuracy_score(ytest, y_val_l, normalize=False)
    print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
    print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
    print("\n")

    if RUN_CROSSVALIDATION:

        for clf, label in zip([log_reg4, radm4, knn4],
                              ['Logistic Regression', 'Random Forest', 'k-NN']):
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            print("Cross-validation: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    print()

    #
    #
    #

    # Analisi dei risultati

    print("╔═══════════════════════╗")
    print("║ Final Accuracy Scores ║")
    print("╚═══════════════════════╝")

    threshold = 0.50

    # --- Logistic Regression ---

    yA1 = log_reg1.predict_proba(X)
    yA2 = log_reg2.predict_proba(X)
    yA3 = log_reg3.predict_proba(X)
    yA4 = log_reg4.predict_proba(X)
    #
    # print(yA1[0])
    # print(yA2[0])
    # print(yA3[0])
    # print(yA4[0])
    #
    # a = yA1[0][1] / (yA1[0][1] + yA2[0][1] + yA3[0][1] + yA4[0][1])
    # b = yA2[0][1] / (yA1[0][1] + yA2[0][1] + yA3[0][1] + yA4[0][1])
    # c = yA3[0][1] / (yA1[0][1] + yA2[0][1] + yA3[0][1] + yA4[0][1])
    # d = yA4[0][1] / (yA1[0][1] + yA2[0][1] + yA3[0][1] + yA4[0][1])
    #
    # print("Normalizzato: %0.2f %0.2f %0.2f %0.2f" % (a, b, c, d))

    y_pred_final_log = np.empty(len(df), dtype=int)
    i = 0
    for elem in y_pred_final_log:
        a = yA1[i][1] / (yA1[i][1] + yA2[i][1] + yA3[i][1] + yA4[i][1])
        b = yA2[i][1] / (yA1[i][1] + yA2[i][1] + yA3[i][1] + yA4[i][1])
        c = yA3[i][1] / (yA1[i][1] + yA2[i][1] + yA3[i][1] + yA4[i][1])
        d = yA4[i][1] / (yA1[i][1] + yA2[i][1] + yA3[i][1] + yA4[i][1])
        m = max(a, b, c, d)
        if m <= threshold:
            y_pred_final_log[i] = 0
        elif m == a:
            y_pred_final_log[i] = 1
        elif m == b:
            y_pred_final_log[i] = 2
        elif m == c:
            y_pred_final_log[i] = 3
        else:
            y_pred_final_log[i] = 4
        i += 1

    # np.set_printoptions(threshold=np.nan)
    # print("Lunghezza y_pred_final: " + str(len(y_pred_final_log)))
    # print(y_pred_final_log)
    # print("Lunghezza y_true: " + str(len(y_true)))
    # print(y_true)

    print("Logistic Regression: %0.2f%%" % (accuracy_score(y_true_int, y_pred_final_log)*100))

    # --- Random Forest ---

    yB1 = radm1.predict_proba(X)
    yB2 = radm2.predict_proba(X)
    yB3 = radm3.predict_proba(X)
    yB4 = radm4.predict_proba(X)

    y_pred_final_rf = np.empty(len(df), dtype=int)
    i = 0
    for elem in y_pred_final_rf:
        a = yB1[i][1] / (yB1[i][1] + yB2[i][1] + yB3[i][1] + yB4[i][1])
        b = yB2[i][1] / (yB1[i][1] + yB2[i][1] + yB3[i][1] + yB4[i][1])
        c = yB3[i][1] / (yB1[i][1] + yB2[i][1] + yB3[i][1] + yB4[i][1])
        d = yB4[i][1] / (yB1[i][1] + yB2[i][1] + yB3[i][1] + yB4[i][1])
        m = max(a, b, c, d)
        if m <= threshold:
            y_pred_final_rf[i] = 0
        elif m == a:
            y_pred_final_rf[i] = 1
        elif m == b:
            y_pred_final_rf[i] = 2
        elif m == c:
            y_pred_final_rf[i] = 3
        else:
            y_pred_final_rf[i] = 4
        i += 1

    # print("Lunghezza y_pred_final: " + str(len(y_pred_final_rf)))
    # print(y_pred_final_rf)
    # print("Lunghezza y_true: " + str(len(y_true)))
    print("Random Forest: %0.2f%%" % (accuracy_score(y_true_int, y_pred_final_rf)*100))

    # --- k-NN ---

    yC1 = knn1.predict_proba(X)
    yC2 = knn2.predict_proba(X)
    yC3 = knn3.predict_proba(X)
    yC4 = knn4.predict_proba(X)

    y_pred_final_knn = np.empty(len(df), dtype=int)
    i = 0
    for elem in y_pred_final_knn:
        a = yC1[i][1] / (yC1[i][1] + yC2[i][1] + yC3[i][1] + yC4[i][1])
        b = yC2[i][1] / (yC1[i][1] + yC2[i][1] + yC3[i][1] + yC4[i][1])
        c = yC3[i][1] / (yC1[i][1] + yC2[i][1] + yC3[i][1] + yC4[i][1])
        d = yC4[i][1] / (yC1[i][1] + yC2[i][1] + yC3[i][1] + yC4[i][1])
        m = max(a, b, c, d)
        if m <= threshold:
            y_pred_final_knn[i] = 0
        elif m == a:
            y_pred_final_knn[i] = 1
        elif m == b:
            y_pred_final_knn[i] = 2
        elif m == c:
            y_pred_final_knn[i] = 3
        else:
            y_pred_final_knn[i] = 4
        i += 1

    # print("Lunghezza y_pred_final: " + str(len(y_pred_final_knn)))
    # print(y_pred_final_knn)
    # print("Lunghezza y_true: " + str(len(y_true)))
    print("k-NN: %0.2f%%" % (accuracy_score(y_true_int, y_pred_final_knn)*100))

    print("\n")

    plt.show()

    print('\nTerminazione del programma.\n')

    # Porzioni di codice ausiliarie alla relazione

    # sklearn.linear_model.LogisticRegression

    # penalty
    # You add a penalty to control properties of the regression coefficients, beyond what the pure likelihood function (i.e. a measure of fit) does.
    # So you optimizie Likelihood+Penalty instead of just maximizing the likelihood.
    # In our case: 'l2', because the ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.

    # Dual
    # Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
    # Prefer dual=False when n_samples > n_features.
    # In our case: False

    # C
    # Regularization artificially discourages complex or extreme explanations of the world even if they fit the what has been observed better.
    # https://www.quora.com/What-is-regularization-in-machine-learning
    # In our case: 1.0, because we didn't find any correlation between the dataset elements.

    # fit_intercept
    # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    # In our case: True, because we noticed that constant added to a decision function can help the program to get better accuracy.

    # intercept_scaling
    # Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.
    # To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
    #  In our case: 1.0, because we didn't wanted to lessen the effect.

    # class_weight
    # Weights associated with classes in the form {class_label: weight}.
    # In our case: not given, so all classes are supposed to have weight one.

    # random_state
    # The seed of the pseudo random number generator to use when shuffling the data. Used only in solvers ‘sag’ and ‘liblinear’.
    # In our case: None, because we already shuffled the data.

    # solver
    # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ is faster for large ones.
    # For multiclass problems, only ‘newton-cg’, ‘sag’ and ‘lbfgs’ handle multinomial loss;
    # In our case: liblinear, because even if the currently used dataset appear to have enough elements to be considered "large",
    # The max_iter was reached which means the coef_ did not converge.
    # In some instances the model may not reach convergence. Nonconvergence of a model indicates that the coefficients are not meaningful
    # because the iterative process was unable to find appropriate solutions; to solve this problem, a slower but more accurated solver
    # (liblinear) gived us the solution to the Convergence Problem.

    # max_iter
    # Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
    # We tried to increase it to solve the Convergence Problem we had while using the sag solver, but we encountered only greater loading times
    # (solver='liblinear', max_iter=100 -> 15~ seconds
    # vs
    # solver='sag', max_iter=500 -> 60~ seconds and convergence problem still unsolved)

    # multi_class
    # Works only for the ‘newton-cg’, ‘sag’ and ‘lbfgs’ solver so we left default ('ovr').

    # warm_start
    # When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
    # It still doesn't support liblinear solver.

    # n_jobs
    # Number of CPU cores used during the cross-validation loop. If given a value of -1, all cores are used.

    # All the information about sklearn.linear_model.LogisticRegression can be read at:
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html





    # sklearn.ensemble.RandomForestClassifier

    # n_estimators
    # The number of trees in the forest.
    #  In our case: 10

    # criterion
    # The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    # Note: this parameter is tree-specific.
    #  In our case: gini

    # max_features
    # The number of features to consider when looking for the best split.
    #  In our case: sqrt(n_features), because we ecountered same result but less time

    # max_depth
    # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
    # min_samples_split samples.
    #  In our case: None, to let maximum freedom to the program

    # min_samples_split
    # The minimum number of samples required to split an internal node:
    #  In our case: 2, the allowed minimum

    # min_samples_leaf
    # The minimum number of samples required to be at a leaf node:
    #  In our case: 1, the allowed minimum

    # min_weight_fraction_leaf
    # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight
    # when sample_weight is not provided.
    #  In our case: 0, the allowed minimum

    # max_leaf_nodes
    # Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number
    # of leaf nodes.
    #  In our case: None, to let maximum freedom to the program

    # min_impurity_split
    # Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
    #  In our case: 1e-7 as default and suggested

    # bootstrap
    # Whether bootstrap samples are used when building trees.
    #  In our case: True, we encountered quite better result

    # oob_score
    # Whether to use out-of-bag samples to estimate the generalization accuracy.
    #  In our case: False, we encountered quite better result

    # n_jobs
    # The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.
    #  In our case: -1

    # random_state
    # If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator;
    # If None, the random number generator is the RandomState instance used by np.random.
    #  In our case: None, because we already shuffled the data.

    # verbose
    # Controls the verbosity of the tree building process.
    #  In our case:

    # warm_start
    # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
    #  In our case: True, we preferred to have new forest each time

    # class_weight
    # “balanced_subsample” or None, optional (default=None) Weights associated with classes in the form {class_label: weight}.
    # If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as
    # the columns of y.
    #  In our case: None (default), because we wanted to leave to each class the same weight, because no one class is neither important or
    # characteristics of another.

    # All the information about sklearn.ensemble.RandomForestClassifier can be read at:
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html





    # KNeighborsClassifier

    # n_neighbors
    # Number of neighbors to use by default for k_neighbors queries.
    # In our case: 5, because we didn't find improvements by increasing the number of neighbors from 5, instead we got only longer loading times.

    # weights : str or callable, optional (default = ‘uniform’)
    # weight function used in prediction.
    # In our case: uniform, because we wanted a uniformed weight to decrease case of uncertainty
    # (but increasing a little the missclassification rate).

    # algorithm
    # Algorithm used to compute the nearest neighbors:
    # In our case: auto,

    # leaf_size
    # Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required
    # to store the tree. The optimal value depends on the nature of the problem.
    # In our case: 30, because we found an optimal point of equilibrium between memory required and loading times.

    # metric
    # the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.
    # In our case:

    # p
    # Power parameter for the Minkowski metric.
    # When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
    # In our case: 1, because we found better resault with the manhattan_distance.

    # metric_params: dict, optional(default=None)
    # Additional keyword arguments for the metric function.
    # In our case:

    # n_jobs
    # The number of parallel jobs to run for neighbors search. If - 1, then the number of jobs is set to the number of CPU cores.
    # Doesn’t affect fit method.
    # In our case: -1

    # All the information about sklearn.neighbors.KNeighborsClassifier can be read at:
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    if RUN_ANALYSIS_LOGISTIC_PENALTY:
        log_reg = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=False, intercept_scaling=1,
                                     class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                     multi_class='ovr',
                                     verbose=0, warm_start=False, n_jobs=-1)
        log_reg.fit(Xtrain, ytrain)
        y_val_l = log_reg.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Logistic Regression Rating_Mature accuracy: ", ris)
        print("Logistic Regression Rating_Mature misclassification: ", ytest.size - mis)

        log_reg = LogisticRegression(penalty='l1', dual=False, C=1.0, fit_intercept=False, intercept_scaling=1,
                                     class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                     multi_class='ovr',
                                     verbose=0, warm_start=False, n_jobs=-1)
        log_reg.fit(Xtrain, ytrain)
        y_val_l = log_reg.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Logistic Regression Rating_Mature accuracy: ", ris)
        print("Logistic Regression Rating_Mature misclassification: ", ytest.size - mis)

        log_reg = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=False, intercept_scaling=1,
                                     class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                     multi_class='ovr',
                                     verbose=0, warm_start=False, n_jobs=-1)
        log_reg.fit(Xtrain, ytrain)
        y_val_l = log_reg.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Logistic Regression Rating_Mature accuracy: ", ris)
        print("Logistic Regression Rating_Mature misclassification: ", ytest.size - mis)

        log_reg = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=False, intercept_scaling=1,
                                     class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                                     multi_class='ovr',
                                     verbose=0, warm_start=False, n_jobs=-1)
        log_reg.fit(Xtrain, ytrain)
        y_val_l = log_reg.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Logistic Regression Rating_Mature accuracy: ", ris)
        print("Logistic Regression Rating_Mature misclassification: ", ytest.size - mis)

    if RUN_ANALYSIS_RANDOMFOREST_NUMBERESTIMATOR:
        radm = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        radm = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        radm = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        radm = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        radm = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        radm = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

    if RUN_ANALYSIS_RANDOMFOREST_ENTROPY:
        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone accuracy: ", ris)
        print("Random Forest Rating_Everyone misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone10+ accuracy: ", ris)
        print("Random Forest Rating_Everyone10+ misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Teen accuracy: ", ris)
        print("Random Forest Rating_Teen misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone accuracy: ", ris)
        print("Random Forest Rating_Everyone misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone10+ accuracy: ", ris)
        print("Random Forest Rating_Everyone10+ misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Teen accuracy: ", ris)
        print("Random Forest Rating_Teen misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

    if RUN_ANALYSIS_RANDOMFOREST_ENTROPYANDNE40:
        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone accuracy: ", ris)
        print("Random Forest Rating_Everyone misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone10+ accuracy: ", ris)
        print("Random Forest Rating_Everyone10+ misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Teen accuracy: ", ris)
        print("Random Forest Rating_Teen misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone accuracy: ", ris)
        print("Random Forest Rating_Everyone misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Everyone10+ accuracy: ", ris)
        print("Random Forest Rating_Everyone10+ misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Teen accuracy: ", ris)
        print("Random Forest Rating_Teen misclassification: ", ytest.size - mis)

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        radm = RandomForestClassifier(n_estimators=40, criterion='entropy', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_split=1e-07, bootstrap=True,
                                      oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                      class_weight=None)
        radm.fit(Xtrain, ytrain)
        y_val_l = radm.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("Random Forest Rating_Mature accuracy: ", ris)
        print("Random Forest Rating_Mature misclassification: ", ytest.size - mis)

    if RUN_ANALYSIS_KNN_NUMBERNEIGHBORS:
        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

    if RUN_ANALYSIS_KNN_LEAFSIZE:
        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=60, p=1,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

    if RUN_ANALYSIS_KNN_P:
        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Everyone10'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Everyone10+ accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Everyone10+ misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Teen'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Teen accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Teen misclassification: ", ytest.size - mis)
        print("\n")

        df2 = df
        y = df2['Rating_Mature'].values
        df2 = df2.drop(['Rating_Everyone'], axis=1)
        df2 = df2.drop(['Rating_Everyone10'], axis=1)
        df2 = df2.drop(['Rating_Teen'], axis=1)
        df2 = df2.drop(['Rating_Mature'], axis=1)
        X = df2.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=1)
        knn.fit(Xtrain, ytrain)
        y_val_l = knn.predict(Xtest)
        ris = accuracy_score(ytest, y_val_l)
        mis = accuracy_score(ytest, y_val_l, normalize=False)
        print("K-Nearest Neighbors Rating_Mature accuracy: ", ris)
        print("K-Nearest Neighbors Rating_Mature misclassification: ", ytest.size - mis)
        print("\n")

