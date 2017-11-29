# Diagnosis of Alzheimer's Disease based on Deep Learning of Neuro Imaging Data

Kwang Bin Lee
Jiayao Wu
Qinwen Huang



Important factors to diagnose the Alzheimer
CDR (Reflecting the presence or absence of dementia). 
The global CDR is based on a standard scoring algorithm which integrates functioning in six individual domains: memory, orientation, judgment and problem solving, community affairs, home and hobbies, and personal care. 

Global CDR scores
0=normal cognition; 
1=mild dementia, 
2=moderate dementia, and 
3=severe dementia. 
CDR 0.5 designates “uncertain dementia” 


MMSE score is less sensitive to cognitive impairment so we are not using it


Data Loading 
--cuda 1 (1 = turn on cuda) (0 = turn off cuda)

--save w (save a model file called w)

--write 1 (1 = write a train.txt) (0 = otherwise)

CommandLine (Example)
python mri_loader.py --write 1 --save w --cuda 0 --epoch 10
