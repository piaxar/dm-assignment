import glob
from numpy import arange
from collections import Counter
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def read_data():
    path = './db/*'
    files = glob.glob(path)
    transactions = []
    for name in files:
        with open(name, 'r') as file:
            transaction = None
            for line in file:
                command = line.strip()
                if command == '**SOF**':
                    transaction = []
                elif command == '**EOF**':
                    transactions.append(transaction)
                else:
                    transaction.append(command)
    return transactions


def min_support_estimator(c: Counter, min_value, max_value, step):
    """Function to estimate num of features in given min_support range"""
    total = sum([a for a in c.values()])

    distr = Counter()
    for i in arange(min_value, max_value, step):
        for value in c.values():
            if value > total * i:
                distr[i] += 1

    return distr


def filter_data(data, values):
    """Removing non-frequent values (values, that are not containing in 'values' list)"""
    return [[x for x in y if x in values] for y in data]


def encode_units(x):
    return 0 if x <= 0 else 1


def main(min_support_threshold=0.0005):
    # reading and counting data
    d = read_data()

    sessions = []
    overall_counter = Counter()

    for s in d:
        sessions.append(Counter(s))

    for s in sessions:
        overall_counter.update(s)

    # estimation of min_support
    # graph = min_support_estimator(overall_counter, 0, 0.002, 0.00001)

    # decided to use min support: 0.05% - > 153 features
    min_sup = (sum([a for a in overall_counter.values()])) * min_support_threshold

    features = []
    for key, value in overall_counter.items():
        if value >= min_sup:
            features.append(key)

    new_data = filter_data(d, features)

    binary_matrix = []
    for transaction in new_data:
        a = Counter({i: 0 for i in features})
        a.update(transaction)
        binary_matrix.append(a)

    df = pd.DataFrame(binary_matrix)

    df.drop(['<1>'], axis=1, inplace=True)
    df.drop(['<2>'], axis=1, inplace=True)
    df.drop(['<3>'], axis=1, inplace=True)
    df.drop(['<4>'], axis=1, inplace=True)

    df = df.applymap(encode_units)

    frequent_itemsets = apriori(df, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules.sort_values(['lift'], ascending=False))
    rules.to_csv("rules.csv")


if __name__ == '__main__':
    main()