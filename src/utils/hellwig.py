import itertools
import math
import pandas as pd


# df = pd.read_csv('Data\\bike_sharing_data.csv')

# potential features
# cols = ['instant', 'holiday', 'workingday',
#         'weathersit', 'temp', 'hum', 'windspeed']


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)

    return pd.Series(diff)


def inverse_difference(last_ob, value):
    # invert differenced forecast
    return value + last_ob


def hellwig(corr: pd.DataFrame, comb):
    info = 0
    for feature in comb:
        d = 0
        for feature_ in comb:
            d += abs(corr[feature][feature_])
        info += (corr[feature]["target_value"]) ** 2 / d
    return info


def run_hellwig(df: pd.DataFrame, cols: list):
    """
    Finds best feature set using hellwig method
    args:
        df : dataframe with values for potential features
        cols : potential feature columns
    """

    # getting all possible combinations of explanatory variables
    combinations = []
    for r in range(1, len(cols) + 1):
        print(len(combinations))
        for combination in itertools.combinations(cols, r):
            combinations.append(combination)

    assert len(combinations) == 2 ** len(cols) - 1

    bestInfo = 0
    bestComb = []
    infos = []
    features = []
    df_corr = df.corr()

    for i, combination in enumerate(combinations):
        print(f"{i}/{len(combinations)}")
        info = hellwig(df_corr, combination)
        if info > bestInfo:
            bestInfo = info
            bestComb = combination
            irrelFeatures = set(cols).difference(combination)
        infos.append(info)
        features.append(combination)

    df = pd.DataFrame({"features": features, "infos": infos})

    with open("hellwigResults.txt", "w") as file:
        for row in df.sort_values(["infos"], ascending=False).iterrows():
            file.write(",".join(row[1]["features"]))
            file.write(f"  -- {row[1]['infos']}\n")

    print(bestInfo)
    print(bestComb)
    print(irrelFeatures)
