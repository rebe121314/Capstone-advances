
#Code gather from 
# https://towardsdatascience.com/drug-discovery-with-deep-learning-under-10-lines-of-codes-742ee306732a

from DeepPurpose import utils,dataset
from DeepPurpose import DTI as models
#The deep purpose library is a collection of deep learning models 
# for drug discovery. 

# Gather data from DAVIS opbtain from the DeepPurpose library
X_drugs, X_targets, y = dataset.load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30)
print('Drug 1: ' + X_drugs[0])
print('Target 1: ' + X_targets[0])
print('Score 1: ' + str(y[0]))


# Encode the data
drug_encoding, target_encoding = 'MPNN', 'CNN'

# Train the model
train, val, test = utils.data_process(X_drugs, X_targets, y, 
                                drug_encoding, target_encoding, 
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)

# Model configuration
config = utils.generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512], 
                         train_epoch = 5, 
                         LR = 0.001, 
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3, 
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )

# Model initialization
model = models.model_initialize(**config)

# Model training
model.train(train, val, test)


# The model can be repurposed to other targets
# For example, we can repurpose the model to SARS-CoV-2 3CL protease
target, target_name = dataset.load_SARS_CoV2_Protease_3CL()
repurpose_drugs, repurpose_drugs_name, repurpose_drugs_pubchem_cid = dataset.load_antiviral_drugs()

y_pred = models.repurpose(X_repurpose = repurpose_drugs, target = target, model = model, 
                          drug_names = repurpose_drugs_name, target_name = target_name, 
                          result_folder = "./result/", convert_y = True)


target, drugs = dataset.load_IC50_1000_Samples()
y_pred = models.virtual_screening(drugs, target, model)

