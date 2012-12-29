HTM
===

Implementation of Numenta's HTM Cortical Learning Algorithm in MATLAB.

See Numenta's whitepaper for a detailed description of the algorithm: 
https://www.numenta.com/htm-overview/education/HTM_CorticalLearningAlgorithms.pdf

Space_Pool.m :    Numenta's spatial pooling algorithm to get sparse distributed representation of input.

Time_Pool.m :   Numenta's temporal pooling algorithm to put SDR representation in temporal context and 
predict future states.

run_HTM.m : Initialize synaptic connections from HTM region's columns to input. Feed samples from periodic
scalar function to the HTM region encoded as sparse distributed representations.
