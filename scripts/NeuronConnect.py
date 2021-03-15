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


def randomization(matrix, neurons):
    # RandomNetwork = matrix
    
    num = 0
    
    while num < 1000:
        # x = random.choice(neurons)
        # y = random.choice(neurons)
        # x2 = random.choice(neurons)
        # y2 = random.choice(neurons)
       
        x = random.randint(0,278)
        y = random.randint(0,278)
        x2 = random.randint(0,278)
        y2 = random.randint(0,278)


        if matrix[x, y] != 0 and matrix[x2, y2] != 0 and x != x2 and y != y2 and x != y and x2 != y2:
        #     #RandomNetwork.replace(RandomNetwork.at[x, y], matrix.at[x2, y2])
            # print(x , " ", y, " ")
            # print(matrix.iloc[x, y])
            # print()
            a = matrix[x2, y2]
            b = matrix[x, y]
            matrix[x, y2] = b
            matrix[x2, y] = a
            # matrix.replace({x: y}, a)
            # matrix.replace({x2, y2}, b)
            num = num + 1
        
    matrixPandas = pd.DataFrame(matrix, index = neurons, columns = neurons)   
    return matrixPandas

        


excelData = pd.read_excel(r'/Users/Adrita1/Programs2.0/Internship/Data/CorrectNeuronConnect.xls') #this location varies for the computer

chart = pd.DataFrame(excelData, columns = ['Neuron 1','Neuron 2', 'Type', 'Nbr'])
chart['Neuron 1'] = chart['Neuron 1'].str.upper()
chart['Neuron 2'] = chart['Neuron 2'].str.upper()

chart = chart.drop(chart[(chart['Neuron 1'] == 'NMJ') | (chart['Neuron 2'] == 'NMJ')].index)
neurons = np.unique(chart[['Neuron 1','Neuron 2']])
#neurons = ["ADAL","ADAR","ADEL","ADER","ADFL","ADFR","ADLL","ADLR","AFDL","AFDR","AIAL","AIAR","AIBL","AIBR","AIML","AIMR","AINL","AINR","AIYL","AIYR","AIZL","AIZR","ALA","ALML","ALMR","ALNL","ALNR","AQR","AS01","AS02","AS03","AS04","AS05","AS06","AS07","AS08","AS09","AS10","AS11","ASEL","ASER","ASGL","ASGR","ASHL","ASHR","ASIL","ASIR","ASJL","ASJR","ASKL","ASKR","AUAL","AUAR","AVAL","AVAR","AVBL","AVBR","AVDL","AVDR","AVEL","AVER","AVFL","AVFR","AVG","AVHL","AVHR","AVJL","AVJR","AVKL","AVKR","AVL","AVM","AWAL","AWAR","AWBL","AWBR","AWCL","AWCR","BAGL","BAGR","BDUL","BDUR","CEPDL","CEPDR","CEPVL","CEPVR","DA01","DA02","DA03","DA04","DA05","DA06","DA07","DA08","DA09","DB01","DB02","DB03","DB04","DB05","DB06","DB07","DD01","DD02","DD03","DD04","DD05","DD06","DVA","DVB","DVC","FLPL","FLPR","HSNL","HSNR","IL1DL","IL1DR","IL1L","IL1R","IL1VL","IL1VR","IL2DL","IL2DR","IL2L","IL2R","IL2VL","IL2VR","LUAL","LUAR","OLLL","OLLR","OLQDL","OLQDR","OLQVL","OLQVR","PDA","PDB","PDEL","PDER","PHAL","PHAR","PHBL","PHBR","PHCL","PHCR","PLML","PLMR","PLNL","PLNR","PQR","PVCL","PVCR","PVDL","PVDR","PVM","PVNL","PVNR","PVPL","PVPR","PVQL","PVQR","PVR","PVT","PVWL","PVWR","RIAL","RIAR","RIBL","RIBR","RICL","RICR","RID","RIFL","RIFR","RIGL","RIGR","RIH","RIML","RIMR","RIPL","RIPR","RIR","RIS","RIVL","RIVR","RMDDL","RMDDR","RMDL","RMDR","RMDVL","RMDVR","RMED","RMEL","RMER","RMEV","RMFL","RMFR","RMGL","RMGR","RMHL","RMHR","SAADL","SAADR","SAAVL","SAAVR","SABD","SABVL","SABVR","SDQL","SDQR","SIADL","SIADR","SIAVL","SIAVR","SIBDL","SIBDR","SIBVL","SIBVR","SMBDL","SMBDR","SMBVL","SMBVR","SMDDL","SMDDR","SMDVL","SMDVR","URADL","URADR","URAVL","URAVR","URBL","URBR","URXL","URXR","URYDL","URYDR","URYVL","URYVR","VA01","VA02","VA03","VA04","VA05","VA06","VA07","VA08","VA09","VA10","VA11","VA12","VB01","VB02","VB03","VB04","VB05","VB06","VB07","VB08","VB09","VB10","VB11","VC01","VC02","VC03","VC04","VC05","VD01","VD02","VD03","VD04","VD05","VD06","VD07","VD08","VD09","VD10","VD11","VD12", "VD13"]

numNeurons = len(neurons)

output = pd.DataFrame(np.zeros((numNeurons, numNeurons)), columns = neurons, index = neurons)

#creating the adjacency matrix (directed, weighted representation)
for i in range(0, len(chart)):
        
        value1 = chart.iloc[i][0]
        value2 = chart.iloc[i][1]
        if value1 == value2:
            continue
        if chart.iloc[i][2] == 'R':
            output.at[value2, value1] += chart.iloc[i][3]
        elif chart.iloc[i][2] == 'Rp':
            output.at[value2, value1] += chart.iloc[i][3]
        elif chart.iloc[i][2] == 'S':      
            output.at[value1, value2] += chart.iloc[i][3]
        elif chart.iloc[i][2] == 'Sp':            
            output.at[value1, value2] += chart.iloc[i][3]
        elif chart.iloc[i][2] == 'EJ':        
            output.at[value1, value2] += chart.iloc[i][3]
            # output.at[value2, value1] = output.at[value2, value1] + chart.iloc[i][3]
outputNumpy = output.to_numpy()
np.save("Output.npy", outputNumpy)
output.to_csv("OutputFixed?.csv")

binaryOutput = pd.DataFrame(np.zeros((numNeurons,numNeurons)), columns = neurons, index = neurons)
index = outputNumpy != 0 
binaryOutput[index] = 1

def BinaryCreate(output, neurons):
    binaryOutput = pd.DataFrame(np.zeros((numNeurons,numNeurons)), columns = neurons, index = neurons)

    #creating binary matrix (still directed)
    for i in range(0, len(chart)):
        if chart.iloc[i][2] == 'R':
                if chart.iloc[i][3] != 0:
                    value1 = chart.iloc[i][0]
                    value2 = chart.iloc[i][1]
                    binaryOutput.at[value2, value1] = 1
                elif chart.iloc[i][3] == 0:
                    value1 = chart.iloc[i][0]
                    value2 = chart.iloc[i][1]
                    binaryOutput.at[value2, value1] = 0
        elif chart.iloc[i][2] == 'Rp':
            if chart.iloc[i][3] != 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 1
            elif chart.iloc[i][3] == 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 0
        elif chart.iloc[i][2] == 'S':
            if chart.iloc[i][3] != 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 1
            elif chart.iloc[i][3] == 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value1, value2] = 0
        elif chart.iloc[i][2] == 'Sp':
            if chart.iloc[i][3] != 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 1
            elif chart.iloc[i][3] == 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 0
        elif chart.iloc[i][2] == 'EJ':
            if chart.iloc[i][3] != 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 1
                binaryOutput.at[value1, value2] = 1
            elif chart.iloc[i][3] == 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                binaryOutput.at[value2, value1] = 0
                binaryOutput.at[value1, value2] = 0


    binaryNum = binaryOutput.to_numpy()
    return binaryNum

#binaryOutput.to_csv("BinaryNeuronOutput.csv")

#create a function with input of neurons and chart
inverseOutput = pd.DataFrame(np.zeros((numNeurons,numNeurons)), columns = neurons, index = neurons)

#creating inverse matrix with inverse values (weighted and directed)
for i in range(0, len(chart)):
    if chart.iloc[i][2] == 'R':
            if chart.iloc[i][3] != 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                inverseOutput.at[value2, value1] = 1/chart.iloc[i][3]
            elif chart.iloc[i][3] == 0:
                value1 = chart.iloc[i][0]
                value2 = chart.iloc[i][1]
                inverseOutput.at[value2, value1] = 0
    elif chart.iloc[i][2] == 'Rp':
        if chart.iloc[i][3] != 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 1/chart.iloc[i][3]
        elif chart.iloc[i][3] == 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 0
    elif chart.iloc[i][2] == 'S':
        if chart.iloc[i][3] != 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 1/chart.iloc[i][3]
        elif chart.iloc[i][3] == 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value1, value2] = 0
    elif chart.iloc[i][2] == 'Sp':
        if chart.iloc[i][3] != 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 1/chart.iloc[i][3]
        elif chart.iloc[i][3] == 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 0
    elif chart.iloc[i][2] == 'EJ':
        if chart.iloc[i][3] != 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 1/chart.iloc[i][3]
            inverseOutput.at[value1, value2] = 1/chart.iloc[i][3]
        elif chart.iloc[i][3] == 0:
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            inverseOutput.at[value2, value1] = 0
            inverseOutput.at[value1, value2] = 0
inverseOutput.to_csv("InverseNeuronOutput.csv")



binaryOutputNumpy = binaryOutput.to_numpy()
inverseOutputNumpy = inverseOutput.to_numpy()

degIn, degOut, degTotal = bct.algorithms.degrees_dir(outputNumpy)
np.save("Degrees.npy", degTotal)

centrality = bct.algorithms.betweenness_wei(outputNumpy)
np.save("Centrality.npy", centrality)

motif1, motif2 = bct.algorithms.motif3funct_bin(binaryOutputNumpy) # make it binary 
np.save("MotifTotalFrequency.npy", motif1)
np.save("MotifFrequencyPerNeuron.npy", motif2)
#npz format for multiple arrays

a,b = bct.algorithms.distance_wei(inverseOutputNumpy)  
np.save("Distance.npy", a) 

d,e,f = bct.algorithms.density_dir(outputNumpy)
np.save("DensityVertices.npy", e)
np.save("DensityEdges.npy", f) 
np.save("TotalDensity.npy", d)

motif3 = bct.algorithms.find_motif34(2, 3)
print(motif3[:,:,0])
print()
motif6 = bct.algorithms.find_motif34(5,3)
print(motif6[:,:,0])
print()
print("matrix of interest: " , bct.algorithms.find_motif34(1,3)[:,:,0])
print("shape: " , np.shape(bct.algorithms.find_motif34(1,3)))
# print(motif2[:,:,0]) #PROBLEM HERE :)
print()
motif3 = bct.algorithms.find_motif34(4,3)
print(motif3[:,:,0])
print()

np.save("Motif3.npy", motif3)
np.save("Motif6.npy", motif6)

# #----------------------------------------------------------------------------------------------------------------------
x = 0

bigThickMatrix = np.zeros((1000,numNeurons,numNeurons))

for x in range (1000):

    random.seed(a = x)

    
    bigThickMatrix[x,:,:] = randomization(outputNumpy, neurons)
    x = x + 1

    print("randomization " , x)

x = 0
np.save("BigMatrix.npy", bigThickMatrix)

#-----------------------------------------------------------------------------------------------------
output.to_csv("NeuronOutputTest.csv")

randomExcel = pd.read_csv (r'/Users/Adrita1/github/neural-connectivity-analysis/data/NeuronOutputTest.csv')
randomChart = pd.DataFrame(randomExcel, index = neurons, columns = neurons)

randomChart.to_csv("RandomChart.csv")



#-----------------------------------------------------------------------------------------------------------------
bigThickMatrix = np.load("BigMatrix.npy")

CoherancemotifMatrix = np.zeros((1000, 13,numNeurons))
FrequencyMotifMatrix = np.zeros((1000, 13, numNeurons))
IntensityMotifMatrix = np.zeros((1000, 13, numNeurons))
for x in range (1000):
    intensity, coherance, frequency = bct.algorithms.motif3funct_wei(bigThickMatrix[x,:,:])
#check motif coherance and intensity and figure out what they mean and do the same thing for the original adjacency
  

    CoherancemotifMatrix[x,:,:] = coherance #CHECK THIS LINE
    FrequencyMotifMatrix[x,:,:] = frequency #CHECK THIS LINE
    IntensityMotifMatrix[x,:,:] = intensity #CHECK THIS LINE
    x = x + 1
    print("iteration " , x)
    
   
np.save("BigMotifCoherance.npy", CoherancemotifMatrix)
np.save("BigMotifIntensity.npy" , FrequencyMotifMatrix)
np.save("BigMotifFrequency.npy" , IntensityMotifMatrix)


np.save("ClusteringCoeffecient.npy", bct.algorithms.clustering_coef_wd(outputNumpy))


BigClusteringCoeff = np.zeros((1000,numNeurons))

for x in range (1000):

    
    BigClusteringCoeff[x,:] = bct.algorithms.clustering_coef_wd(bigThickMatrix[x,:,:])
    x = x + 1

    print("iteration " , x)



np.save("BigClusteringMatrix.npy", BigClusteringCoeff)


WeightedMotifIntensity, WeightedMotifCoherance, WeightedMotifFrequency = bct.algorithms.motif3funct_wei(outputNumpy)
np.save("WeightedMotifIntensity.npy", WeightedMotifIntensity)
np.save("WeightedMotifCoherance.npy", WeightedMotifCoherance)
np.save("WeightedMotifFrequency.npy", WeightedMotifFrequency)


# #optional assignments: length of connection (function for physiological lengths)
# #optional assignment: stats test on my own, randomization function on toolbox, networkx

# np.save("P-ValuesCoherance.npy", scipy.stats.ttest_1samp(BigCoheranceMatrix, coherance, axis=0))

#use matplotlib for histograms

# NetworkGraph = nx.Graph(outputNumpy) #MAKE THIS TAKE IN AN UNDIRECTED GRAPH

# NetworkUndirected = NetworkGraph.to_undirected( as_view=False)

# LargestConnected = nx.to_numpy_matrix(max((NetworkUndirected.subgraph(c) for c in nx.connected_components(NetworkUndirected)), key=len))

# print("Numpy: ", np.shape(outputNumpy))
# print(np.shape(LargestConnected))
# # nx.draw(nx.DiGraph(LargestConnected))
# # plt.draw()
# # plt.show()

# # print("out degrees: " , np.nonzero(np.all(outputNumpy == 0, axis=0)))
# # print("in degrees: ", np.nonzero(np.all(outputNumpy == 0, axis=1)))

# print("recieving: " , sum(outputNumpy[:,4]==0))
# print("sending: " , sum(outputNumpy[4,:]==0))


# bigThickMatrix = np.load("BigMatrix.npy")

# x = 0
# BigCoheranceMatrix = np.zeros((1000,13,numNeurons))

# for x in range (1000):

#     intensity, coherance, frequency = bct.algorithms.motif3funct_wei(bigThickMatrix[x,:,:])
#     BigCoheranceMatrix[x,:,:] = coherance
#     x = x + 1

# np.save("BigCoheranceMatrix.npy", BigCoheranceMatrix)
# # BigCoheranceMatrix = np.load("BigCoheranceMatrix.npy")


# BigCoherance = pd.DataFrame(BigCoheranceMatrix[:,:,4])
# BigCoherance.to_csv("BigCoheranceMatrix.csv")

