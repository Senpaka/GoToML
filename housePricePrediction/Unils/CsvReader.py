import csv

def readDataset(url):
    csv_dict = {}
    with open(url, mode="r") as csv_file:
        csv_riader = csv.DictReader(csv_file)


readDataset("/Users/senpaka/PycharmProjects/learn_ml/housePricePrediction/datasets/housing_dataset.csv")