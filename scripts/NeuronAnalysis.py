import numpy as np
import pandas as pd
import bct
import random
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def testing_sample(filename): 

    #------------------------------------------------------------------------------
    # Testing Sample (5x5 sample)
    # input: '/Users/Adrita1/Programs2.0/Internship/Data/Test.xls'
    # output: intensity, coherance, frequency of 5x5 array
    #------------------------------------------------------------------------------
    #TESTING BEGINS HERE
    testData = pd.read_excel(filename, header=None) #this location varies for the computer
    #testPandas = pd.DataFrame(testData)
    testDataNumpy = testData.to_numpy()

    print(testDataNumpy.shape)
    testIntensity, testCoherance, testFrequency = bct.algorithms.motif3funct_wei(testDataNumpy.T)
    np.save("testIntensity.npy", testIntensity)
    np.save("testCoherance.npy", testCoherance)
    np.save("testFrequency.npy", testFrequency)
    print("Saved Successfully")

    #TESTING ENDS HERE

def print_motif_arrays():

    #------------------------------------------------------------------------------
    # Printing motif arrays
    # input: none
    # output: 13 3x3 arrays 
    #------------------------------------------------------------------------------
    motif1 = bct.algorithms.find_motif34(1,3)
    print(motif1[:,:,0])
    motif2 = bct.algorithms.find_motif34(2,3)
    print(motif2[:,:,0])
    motif3 = bct.algorithms.find_motif34(3,3)
    print(motif3[:,:,0])
    motif4 = bct.algorithms.find_motif34(4,3)
    print(motif4[:,:,0])
    motif5 = bct.algorithms.find_motif34(5,3)
    print(motif5[:,:,0])
    motif6 = bct.algorithms.find_motif34(6,3)
    print(motif6[:,:,0])
    motif7 = bct.algorithms.find_motif34(7,3)
    print(motif7[:,:,0])
    motif8 = bct.algorithms.find_motif34(8,3)
    print(motif8[:,:,0])
    motif9 = bct.algorithms.find_motif34(9,3)
    print(motif9[:,:,0])
    motif10 = bct.algorithms.find_motif34(10,3)
    print(motif10[:,:,0])
    motif11 = bct.algorithms.find_motif34(11,3)
    print(motif11[:,:,0])
    motif12 = bct.algorithms.find_motif34(12,3)
    print(motif12[:,:,0])
    motif13 = bct.algorithms.find_motif34(13,3)
    print(motif13[:,:,0])

def networkx_celegans(filename):
    #------------------------------------------------------------------------------
    # Creating NetworkX Graph for C. Elegans connectome 
    # input: C. Elegans network (278x278) : '/Users/Adrita1/Programs2.0/Internship/Data/CorrectNeuronConnect.xls'
    # output: NetworkXGraph 
    #------------------------------------------------------------------------------
    # 
    excelData = pd.read_excel(filename) #this location varies for the computer

    chart = pd.DataFrame(excelData, columns = ['Neuron 1','Neuron 2', 'Type', 'Nbr'])
    chart['Neuron 1'] = chart['Neuron 1'].str.upper()
    chart['Neuron 2'] = chart['Neuron 2'].str.upper()

    chart = chart.drop(chart[(chart['Neuron 1'] == 'NMJ') | (chart['Neuron 2'] == 'NMJ')].index)
    neurons = np.unique(chart[['Neuron 1','Neuron 2']])
    numNeurons = len(neurons)


    outputNumpy = np.load("Output.npy")
    NetworkGraph = nx.Graph(outputNumpy) #MAKE THIS TAKE IN AN UNDIRECTED GRAPH

    NetworkUndirected = NetworkGraph.to_undirected( as_view=False)

    LargestConnected = nx.to_numpy_matrix(max((NetworkUndirected.subgraph(c) for c in nx.connected_components(NetworkUndirected)), key=len))

    print("Numpy: ", np.shape(outputNumpy))
    print(np.shape(LargestConnected))
    # nx.draw(nx.DiGraph(LargestConnected))
    # plt.draw()
    # plt.show()

    # print("out degrees: " , np.nonzero(np.all(outputNumpy == 0, axis=0)))
    # print("in degrees: ", np.nonzero(np.all(outputNumpy == 0, axis=1)))

    print("recieving: " , sum(outputNumpy[:,4]==0))
    print("sending: " , sum(outputNumpy[4,:]==0))
    return numNeurons

def initialize(): 
    #------------------------------------------------------------------------------
    # Loading in necessary datasets 
    # input: none
    # output: Random 1000x279x279 matrix, BigCoherancematrix, WeightedMotifCoherance 
    #------------------------------------------------------------------------------
    bigThickMatrix = np.load("BigMatrix.npy")


    BigCoheranceMatrix = np.load("BigCoheranceMatrix.npy")


    #These three variables are together
    BigCoheranceMatrix = np.load("BigMotifCoherance.npy")
    BigMotifIntensity = np.load("BigMotifIntensity.npy")
    BigMotifFrequency = np.load("BigMotifFrequency.npy")
    # print("Motif Frequency: ", MotifFrequency[:, 4])

    # print(WeightedMotifIntensity[:,4])

    WeightedMotifCoherance = np.load("WeightedMotifCoherance.npy")

    big_coherance = BigCoheranceMatrix[:, 0, 0]
    # print(x)
    print(np.shape(BigCoheranceMatrix))
    return big_coherance, WeightedMotifCoherance, BigCoheranceMatrix

def mann_whitney_u(big_coherance, WeightedMotifCoherance, BigCoheranceMatrix, numNeurons):
    #------------------------------------------------------------------------------
    # Performing Mann Whitney U test and plotting results
    # input: Random Motif Coherance, Actual Motif Coherance
    # output: P Values for test 
    #------------------------------------------------------------------------------
    statistic, p_value = scipy.stats.mannwhitneyu(big_coherance, [WeightedMotifCoherance[0,0]], use_continuity=True, alternative='two-sided')
    # print(p_value)

    #plot the p values for all the motifs/neurons and also a histogram per motif (with the p values)

    SummaryPValues = np.empty((13,numNeurons))
    SummaryPValues[:] = np.nan

    for row in range(13):
        # x = BigCoheranceMatrix[:, row, col]
        # statistic, p_value = scipy.stats.mannwhitneyu([WeightedMotifCoherance[row,col]], x, use_continuity=True, alternative='two-sided')
        # SummaryPValues[row, col] = p_value
        #print(WeightedMotifFrequency[row, 278])
        for col in range(numNeurons):
            x = BigCoheranceMatrix[:, row, col]
            if len(np.unique(x)) == 1:
                continue
            # print("row: ", row)
            # print("col: ", col)
            # print(x)
            # print(WeightedMotifCoherance[row,col])
            statistic, p_value = scipy.stats.mannwhitneyu(x, [WeightedMotifCoherance[row,col]], use_continuity=True, alternative='two-sided')
            SummaryPValues[row, col] = p_value

    return SummaryPValues

def graphing_statistics(SummaryPValues):
    #------------------------------------------------------------------------------
    # Graphing statistics results
    # input: P values
    # output: plt graph 
    #------------------------------------------------------------------------------   
    SummaryPValuesDF = pd.DataFrame(SummaryPValues)
    SummaryPValuesDF.to_csv("SummaryPValues.csv")
    n, bins, patches = plt.hist(SummaryPValues, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('P Values')
    plt.ylabel('Frequency')
    plt.title('Frequency of P Values for Coherance (Mann Whitney U Test)')

    # plt.xlim(0, 4)
    # plt.ylim(0, 700)
    plt.grid(True)
    plt.show()