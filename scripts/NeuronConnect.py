import numpy as np
import pandas as pd
import bct
import random
import networkx as nx
import matplotlib.pyplot as plt



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
            matrix[x, y] = a
            matrix[x2, y2] = b
            # matrix.replace({x: y}, a)
            # matrix.replace({x2, y2}, b)
            num = num + 1
        
    matrixPandas = pd.DataFrame(matrix, index = neurons, columns = neurons)   
    return matrixPandas

        


excelData = pd.read_excel (r'C:\Users\Adrita\Downloads\CorrectNeuronConnect.xlsx') #this location varies for the computer

chart = pd.DataFrame(excelData, columns = ['Neuron 1','Neuron 2', 'Type', 'Nbr'])


neurons = ["ADAR","ADAL","ADEL","ADER","ADFL","ADFR","ADLL","ADLR","AFDL","AFDR","AIAL","AIAR","AIBL","AIBR","AIML","AIMR","AINL","AINR","AIYL","AIYR","AIZL","AIZR","ALA","ALML","ALMR","ALNL","ALNR","AQR","AS01","AS02","AS03","AS04","AS05","AS06","AS07","AS08","AS09","AS10","AS11","ASEL","ASER","ASGL","ASGR","ASHL","ASHR","ASIL","ASIR","ASJL","ASJR","ASKL","ASKR","AUAL","AUAR","AVAL","AVAR","AVBL","AVBR","AVDL","AVDR","AVEL","AVER","AVFL","AVFR","AVG","AVHL","AVHR","AVJL","AVJR","AVKL","AVKR","AVL","AVM","AWAL","AWAR","AWBL","AWBR","AWCL","AWCR","BAGL","BAGR","BDUL","BDUR","CEPDL","CEPDR","CEPVL","CEPVR","DA01","DA02","DA03","DA04","DA05","DA06","DA07","DA08","DA09","DB01","DB02","DB03","DB04","DB05","DB06","DB07","DD01","DD02","DD03","DD04","DD05","DD06","DVA","DVB","DVC","FLPL","FLPR","HSNL","HSNR","IL1DL","IL1DR","IL1L","IL1R","IL1VL","IL1VR","IL2DL","IL2DR","IL2L","IL2R","IL2VL","IL2VR","LUAL","LUAR","OLLL","OLLR","OLQDL","OLQDR","OLQVL","OLQVR","PDA","PDB","PDEL","PDER","PHAL","PHAR","PHBL","PHBR","PHCL","PHCR","PLML","PLMR","PLNL","PLNR","PQR","PVCL","PVCR","PVDL","PVDR","PVM","PVNL","PVNR","PVPL","PVPR","PVQL","PVQR","PVR","PVT","PVWL","PVWR","RIAL","RIAR","RIBL","RIBR","RICL","RICR","RID","RIFL","RIFR","RIGL","RIGR","RIH","RIML","RIMR","RIPL","RIPR","RIR","RIS","RIVL","RIVR","RMDDL","RMDDR","RMDL","RMDR","RMDVL","RMDVR","RMED","RMEL","RMER","RMEV","RMFL","RMFR","RMGL","RMGR","RMHL","RMHR","SAADL","SAADR","SAAVL","SAAVR","SABD","SABVL","SABVR","SDQL","SDQR","SIADL","SIADR","SIAVL","SIAVR","SIBDL","SIBDR","SIBVL","SIBVR","SMBDL","SMBDR","SMBVL","SMBVR","SMDDL","SMDDR","SMDVL","SMDVR","URADL","URADR","URAVL","URAVR","URBL","URBR","URXL","URXR","URYDL","URYDR","URYVL","URYVR","VA01","VA02","VA03","VA04","VA05","VA06","VA07","VA08","VA09","VA10","VA11","VA12","VB01","VB02","VB03","VB04","VB05","VB06","VB07","VB08","VB09","VB10","VB11","VC01","VC02","VC03","VC04","VC05","VD01","VD02","VD03","VD04","VD05","VD06","VD07","VD08","VD09","VD10","VD11","VD12", "VD13"]


output = pd.DataFrame(np.zeros((279,279)), columns = neurons, index = neurons)

#creating the adjacency matrix (directed, weighted representation)
for i in range(0, len(chart)):
        
        if chart.iloc[i][2] == 'R':
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            output.at[value2, value1] = output.at[value2, value1] + chart.iloc[i][3]
        elif chart.iloc[i][2] == 'Rp':
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            output.at[value2, value1] = output.at[value2, value1] + chart.iloc[i][3]
        elif chart.iloc[i][2] == 'S':
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            output.at[value1, value2] = output.at[value2, value1] + chart.iloc[i][3]
        elif chart.iloc[i][2] == 'Sp':
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
            output.at[value1, value2] = output.at[value2, value1] + chart.iloc[i][3]
        elif chart.iloc[i][2] == 'EJ':
            value1 = chart.iloc[i][0]
            value2 = chart.iloc[i][1]
           
            output.at[value1, value2] = output.at[value2, value1] + chart.iloc[i][3]
            output.at[value2, value1] = output.at[value2, value1] + chart.iloc[i][3]


def BinaryCreate(output, neurons):
    binaryOutput = pd.DataFrame(np.zeros((279,279)), columns = neurons, index = neurons)

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
inverseOutput = pd.DataFrame(np.zeros((279,279)), columns = neurons, index = neurons)

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



outputNumpy = output.to_numpy()
# binaryOutputNumpy = binaryOutput.to_numpy()
# inverseOutputNumpy = inverseOutput.to_numpy()

# degIn, degOut, degTotal = bct.algorithms.degrees_dir(outputNumpy)
# np.save("Degrees.npy", degTotal)

# centrality = bct.algorithms.betweenness_wei(outputNumpy)
# np.save("Centrality.npy", centrality)

# motif1, motif2 = bct.algorithms.motif3funct_bin(binaryOutputNumpy) # make it binary 
# np.save("MotifTotalFrequency.npy", motif1)
# np.save("MotifFrequencyPerNeuron.npy", motif2)
# #npz format for multiple arrays

# a,b = bct.algorithms.distance_wei(inverseOutputNumpy)  
# np.save("Distance.npy", a) 

# d,e,f = bct.algorithms.density_dir(outputNumpy)
# np.save("DensityVertices.npy", e)
# np.save("DensityEdges.npy", f) 
# np.save("TotalDensity.npy", d)

# motif3 = bct.algorithms.find_motif34(2, 3)
# print(motif3[:,:,0])
# print()
# motif6 = bct.algorithms.find_motif34(5,3)
# print(motif6[:,:,0])
# print()
# motif2 = bct.algorithms.find_motif34(0,3)
# print(motif2[:,:,0])
# print()
# motif3 = bct.algorithms.find_motif34(4,3)
# print(motif3[:,:,0])
# print()

# np.save("Motif3.npy", motif3)
# np.save("Motif6.npy", motif6)



# output.to_csv("NeuronOutputTest.csv")

# randomExcel = pd.read_csv (r'C:\Users\Adrita\Programs2.0\Internship\NeuronOutputTest.csv')
# randomChart = pd.DataFrame(randomExcel, index = neurons, columns = neurons)

# randomChart.to_csv("RandomChart.csv")
#-----------------------------------------------------------------------------------------------------------------
# x = 0

# bigThickMatrix = np.zeros((1000,279,279))

# for x in range (1000):

#     random.seed(a = x)

    
#     bigThickMatrix[x,:,:] = randomization(outputNumpy, neurons)
#     x = x + 1

#     print("randomization " , x)
#     #save this into a 3d matrix (278x278x1000)
#     #add the random graphs into the third dimension

# x = 0
# np.save("BigMatrix.npy", bigThickMatrix)
# motifMatrix = np.zeros((13,1000))

# for x in range (1000):
#     intensity, coherance, frequency = bct.algorithms.motif3funct_wei(bigThickMatrix[x,:,:])
# #check motif coherance and intensity and figure out what they mean and do the same thing for the original adjacency
  

#     motifMatrix[:,x] = frequency #CHECK THIS LINE
#     x = x + 1
#     print("iteration " , x)
    
   
# np.save("BigMotifFrequency.npy", motifMatrix)
#----------------------------------------------------------------------------------------------------------------------
intensity, coherance, frequency = bct.algorithms.motif3funct_wei(outputNumpy)
np.save("WeightedMotifFrequency.npy", frequency)
#optional assignments: length of connection (function for physiological lengths)
#optional assignment: stats test on my own, randomization function on toolbox, networkx




NetworkGraph = nx.DiGraph(outputNumpy)
nx.draw(NetworkGraph, with_labels=True, font_weight='bold')
plt.show()