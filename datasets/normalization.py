#!/usr/bin/python3

dataFile = open(input("Enter filepath to data array(s) here:"), "r")
outputFile = open(input("Enter filepath to write normalized data:"), "w")

for line in dataFile:
    dataList = [float(i) for i in line.split(",")]

    minimum = min(dataList)
    domain = max(dataList) - minimum

    if domain == 0: continue

    normedDataList = [(i-minimum)/(domain) for i in dataList]

    outputFile.write("{0}\n".format(",".join([str(i) for i in normedDataList])))

dataFile.close()
outputFile.close()