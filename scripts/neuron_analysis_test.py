from NeuronAnalysis import print_motif_arrays
from NeuronAnalysis import testing_sample
from NeuronAnalysis import networkx_celegans
from NeuronAnalysis import initialize
from NeuronAnalysis import mann_whitney_u
from NeuronAnalysis import graphing_statistics

print_motif_arrays()

testing_sample('/Users/Adrita1/Programs2.0/Internship/Data/Test.xls')

num_neurons = networkx_celegans('/Users/Adrita1/Programs2.0/Internship/Data/CorrectNeuronConnect.xls')

big_coherance, weighted_motif_coherance, big_coherance_matrix = initialize()

summary_p_values = mann_whitney_u(big_coherance, weighted_motif_coherance, big_coherance_matrix, num_neurons)

graphing_statistics(summary_p_values)