from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData,buildExamplesFromXORData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

XORData = buildExamplesFromXORData()
def testXORData(hiddenLayers = [16]):
    return buildNeuralNet(XORData,maxItr = 200, hiddenLayerList =  hiddenLayers)

def stddev(values):
    mean = sum(values)/float(len(values))
    differencetotal = 0
    for value in values:
        differencetotal += abs(value - mean)**2
    differencetotal /= float(len(values))
    return sqrt(differencetotal)

# def q5():
#     print 'q5'
#     pen = []
#     car = []
#     for i in range(5):
#         pen.append(testPenData()[1])
#         print("%d times pen DONE" % (i + 1))
#         car.append(testCarData()[1])
#         print("%d times car DONE" % (i + 1))
#
#
#     print('Pen data set:')
#     print("Max %s | Avg %s | STD %s" % (max(pen), average(pen), stDeviation(pen)))
#     print('Car data set:')
#     print("Max %s | Avg %s | STD %s" % (max(car), average(car), stDeviation(car)))

# def q6():
#     print 'q6'
#     for y in range(0, 41, 5):
#         pen = []
#         car = []
#         for x in range(5):
#             print "Pen test number:", x + 1, "y:", y
#             result = testPenData([y])
#             pen.append(result[1])
#             print "Car test number:", x + 1, "y:", y
#             result = testCarData([y])
#             car.append(result[1])
#         print("Pen Data Avg, Max, Stdev: " + str((sum(pen) / float(len(pen)))) + ", " + str(max(pen)) + ", " + str(
#             stddev(pen)) + "\n")
#         print("Car Data Avg, Max, Stdev: " + str((sum(car) / float(len(car)))) + ", " + str(max(car)) + ", " + str(
#             stddev(car)) + "\n")

#q5()
#q6()