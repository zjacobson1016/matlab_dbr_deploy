# Databricks notebook source
#Good refs
#https://github.com/nicknochnack/DeepAudioClassification/blob/main/AudioClassification.ipynb
#https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb
#https://www.kaggle.com/competitions/PLAsTiCC-2018
#https://github.com/databricks-industry-solutions/utilities-cv

# COMMAND ----------

# MAGIC %md
# MAGIC #Paralleize tensor flow on spark
# MAGIC #https://github.com/tensorflow/ecosystem/blob/master/spark/spark-tensorflow-distributor/examples/simple/example.py

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
import tensorflow as tf
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.utils import to_categorical
import mlflow
import pandas as pd
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

#Orginal mapping from expore data 
fault_types_orgi = {
    '0000': 'No Fault',
    '1000': 'Single Line to Ground A',
    '0100': 'Single Line to Ground B',
    '0010': 'Single Line to Ground C',
    '0011': 'Line-to-Line BC',
    '0101': 'Line-to-Line AC',
    '1001': 'Line-to-Line AB',
    '1010': 'Line-to-Line with Ground AB',
    '0101': 'Line-to-Line with Ground AC',
    '0110': 'Line-to-Line with Ground BC',
    '0111': 'Three-Phase',
    '1111': 'Three-Phase with Ground',
    '1011': 'Line A Line B to Ground Fault'
}

# COMMAND ----------

#Map Fault codes to String Values. This will be our new target column
import pandas as pd
fault_types = {
    'z_z': "1",
    'a_c_g':"2",
    'c_g':"3",
    'a_g':"4",
    'a_b_c_g':"5",
    'b_g':"6",
    'a_b_g':"7",
    'b_c_g':"8"
}

@F.pandas_udf(StringType())
def map_codes(codes:pd.Series) -> pd.Series:
  return codes.map(fault_types)

# COMMAND ----------

spark.sql('select * from main.zach_jacobson.fault_simulation').display()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import re

regex_pattern = r"(?<=_)[a-zA-Z]+"

# Define a UDF to extract all matching patterns
def extract_consecutive_letters(text):
    # Match letters that are preceded and followed by an underscore
    return re.findall(regex_pattern, text)

# Register the UDF with the appropriate return type
extract_udf = udf(extract_consecutive_letters, ArrayType(StringType()))

# Apply the UDF to add a new column with extracted sequences
df_fault_transformation = spark.sql('select * from main.zach_jacobson.fault_simulation') \
.withColumn('fault_class',F.when(F.col("fault_class")=="unfaulted","CB0_0_z_z").otherwise(F.col("fault_class"))) \
.withColumn('Bus_loc_extr', F.split(F.col('fault_class'), "_")) \
.withColumn('Bus_loc', F.expr("try_element_at(Bus_loc_extr, 2)")) \
.withColumn("extracted_letters", F.concat_ws("_",extract_udf(F.col("fault_class")))) \
.drop("Bus_loc_extr")


df_fault_transformation.display()

# COMMAND ----------

from pyspark.sql.types import IntegerType

inputs_pd = df_fault_transformation \
.withColumn('fault_type_coded',map_codes(F.col('extracted_letters')).cast(IntegerType())) \
.withColumn('id_temp',F.monotonically_increasing_id())


# COMMAND ----------

inputs_pd.groupBy('fault_type_coded').agg(F.count(F.col('fault_type_coded'))).display()

# COMMAND ----------

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

batch_size = 554
#Split out dataframe into simulation batches. 40 simulations, produces 550 records per match. Each simulation is placed at a 55.5 second threshold

def add_batch_id(pdf):
    # Calculate number of rows
    num_rows = len(pdf)
    # Create the batch_id column
    #pdf = pdf.sample(frac=2,replace=True).reset_index(drop=True)
    pdf['batch_id'] = (pd.Series(range(num_rows)) // batch_size) + 1
    return pdf

# Apply UDF to DataFrame, grouping by 'id' for demonstration purposes
# Since we are batching by fixed sizes rather than by ID, this example doesn't use the 'id' grouping directly for batch calculation.
df_with_batches = inputs_pd.groupBy().applyInPandas(add_batch_id,schema="V1mag_13 double,V1ph_13 double,V2mag_13 double, V2ph_13 double,V0mag_13 double,V0ph_13 double,I1mag_1 double,I1ph_1 double,I2mag_1 double,I2ph_1 double,I0mag_1 double,I0ph_1 double,fault_class string,Bus_loc string,extracted_letters string,fault_type_coded integer,id_temp long,batch_id integer").filter(F.col('batch_id')!= '41')

# Show result
df_with_batches.sort("batch_id").display()

# COMMAND ----------

df_with_batches.count()

# COMMAND ----------

columns_to_round = ['V1mag_13','V1ph_13','V2mag_13','V2ph_13','V0mag_13','V0ph_13','I1mag_1','I1ph_1','I2mag_1','I2ph_1','I0mag_1','I0ph_1']
df_with_batches_round = df_with_batches.select('batch_id','fault_type_coded','extracted_letters','Bus_loc',*[F.round(F.col(column), 3).alias(column) for column in columns_to_round])

# COMMAND ----------

df_with_batches_round.groupBy('batch_id').count().display()

# COMMAND ----------

### For Voltage Variables
inputs_pd_pandas = df_with_batches_round.toPandas()
plt.figure(figsize=(22, 5))
plt.plot(inputs_pd_pandas["V1mag_13"], 'r', label='Va')
#plt.plot(inputs_pd_pandas["V1ph_13"], 'b', label='Vb')
#plt.plot(inputs_pd_pandas["V1mag_13"], 'g', label='Vc')

# Add legend to the plot
plt.legend()

# Set plot labels and title
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage Over Time with Line Colors')

# Show the plot
plt.show()

# COMMAND ----------

from pyspark.sql import functions as F
#Here the class labels are pivoted to count occurences of each class per column. We will then create proabailities for each waveform/cycle (i.e. batch_id)
df_test = df_with_batches_round.groupBy('batch_id').pivot("fault_type_coded").count().fillna(0)

#Creating probabilities for each fault type class
df_target = df_test.withColumn('batch_class_total',F.col('1')+F.col('2')+F.col('3')+F.col('4')+F.col('5')+F.col('6')+F.col('7')+F.col('8')) \
.withColumn('fault_type',F.array(F.col('1')/F.col('batch_class_total'),F.col('2')/F.col('batch_class_total'),F.col('3')/F.col('batch_class_total'),F.col('4')/F.col('batch_class_total'),F.col('5')/F.col('batch_class_total'),F.col('6')/F.col('batch_class_total'),F.col('7')/F.col('batch_class_total'),F.col('8')/F.col('batch_class_total'))) \
.select('batch_id','fault_type')

# COMMAND ----------

df_test.display()
df_target.display()

# COMMAND ----------

df = df_with_batches.groupBy(F.col('batch_id')).agg(F.collect_list(F.array(['V1mag_13',
 'V1ph_13',
 'V2mag_13',
 'V2ph_13',
 'V0mag_13',
 'V0ph_13',
 'I1mag_1',
 'I1ph_1',
 'I2mag_1',
 'I2ph_1',
 'I0mag_1',
 'I0ph_1'])).alias('features')) \
.join(df_target,['batch_id'],'left')

# COMMAND ----------

df.display()

# COMMAND ----------

#Check to make sure each voltage array row is the same size. This is crucial for our input 3-D tensor to work later.
df.withColumn('feature_size',F.size('features')) \
.select('feature_size') \
.display()

# COMMAND ----------

df_locations = spark.sql('select * from main.zach_jacobson.fault_locations') \
.withColumn('distance_sub_station',F.sqrt(F.pow(F.col('X_Coord'),2)+F.pow(F.col('Y_Coord'),2)))
df_locations.display()
inputs_pd.display()

# COMMAND ----------

#Shape I want to get into
x = df.select("features").rdd.map(lambda r: r[0]).collect()  # python list
features = np.array(x)

y = df.select("fault_type").rdd.map(lambda r: r[0]).collect() # python list
labels = np.array(y)
#labels = to_categorical(np.array(y)-1)


# COMMAND ----------

features.shape

# COMMAND ----------

#https://github.com/CSCfi/pytorch-ddp-examples/blob/master/run-ddp-gpu4-mlflow.sh

# COMMAND ----------

# MAGIC %pip install lightning

# COMMAND ----------

# MAGIC %md
# MAGIC #Resources:
# MAGIC ##-https://github.com/databricks-industry-solutions/utilities-cv/blob/main/03_MultiGPU_ModelTraining.py
# MAGIC ##-https://github.com/CSCfi/pytorch-ddp-examples/blob/master/mnist_lightning_ddp.py
# MAGIC ##-https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_CNN.py
# MAGIC ##-https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html
# MAGIC ##-https://www.youtube.com/watch?v=eQvI5eAL0nA
# MAGIC ##-https://github.com/mlflow/mlflow/blob/master/examples/pytorch/MNIST/mnist_autolog_example.py

# COMMAND ----------

# MAGIC %md
# MAGIC #Definitions
# MAGIC #DataLoader: Dataloaders are iterables over the dataset. So when you iterate over it, it will return B randomly from the dataset collected samples (including the data-sample and the target/label), where B is the batch-size
# MAGIC #Pytorch_lighting.Trainer: 
# MAGIC ## -Automatically enabling/disabling grads,
# MAGIC ## -Running the training,validation and test dataloaders,
# MAGIC ## -Calling the Callbacks at the appropriate times,
# MAGIC ## -Putting batches and computations on the correct devices
# MAGIC #TorchDistributor: A class to support distributed training on PyTorch and PyTorch Lightning using PySpark.
# MAGIC #Lighting Required/Core Methods:https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

# COMMAND ----------

import torch
from torch import optim, nn, utils, Tensor
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pyspark.ml.torch.distributor import TorchDistributor
import mlflow
import os

NUM_WORKERS = 2
# NOTE: This assumes the driver node and worker nodes have the same instance type.
NUM_GPUS_PER_WORKER = torch.cuda.device_count() # CHANGE AS NEEDED
USE_GPU = NUM_GPUS_PER_WORKER > 0
 
username = spark.sql("SELECT current_user()").first()['current_user()']

experiment_path = f'/Users/{username}/pytorch-distributor'

# This is needed for later in the notebook
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import argparse
import os
import torch
import torch.nn.functional as F
#from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
#import lightning as L
import pytorch_lightning as pL
import mlflow
import sys

class MnistDataModule(pL.LightningDataModule):

  def __init__(self,dataset_object, batch_size:int=554):
    super().__init__()
    self.batch_size = batch_size
    self.dataset_object= dataset_object

  def setup(self, stage:str):
    mnist_full = self.dataset_object
    self.mnist_test = self.dataset_object[2]
    self.mnist_predict = self.dataset_object[2]
    self.mnist_train, self.mnist_val = (self.dataset_object[0],self.dataset_object[1])

  def train_dataloader(self):
    return utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

  def val_dataloader(self):
    return utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
    return utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

  def predict_dataloader(self):
    return utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size)

# COMMAND ----------

from torch import optim, nn, utils
from torchvision import datasets, transforms
import pytorch_lightning as pl
import mlflow
  

class Conv1DFaultModel(pL.LightningModule):
    import argparse
    import os
    import torch
    import torch.nn.functional as F
    #from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    #import lightning as L
    import pytorch_lightning as L
    import mlflow
    import sys
    from torch.utils.data import DataLoader

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None
        self.conv1d = nn.Conv1d(12, 32, 3,1,padding=3//2)
        self.dense1 = nn.Linear(277*32, 120) #hard code for now can think of better way to structure this
        self.dense2 = nn.Linear(120, 60)
        self.dense3 = nn.Linear(60, 8) #... continue to add layers for better performance

    def forward(self, input_data):
        output = F.relu(self.conv1d(input_data))
        #output = self.relu(output)
        #output = self.dropout(output)
        output =F.max_pool1d(output,2,2)
        #output = self.flatten(output)
        #output = self.dense1(output)
        shape1 = output.shape[1] #dimension 1 to flatten out for fully connected nn layer
        shape2 = output.shape[2] #dimension 1 to flatten out for fully connected nn layer
        output = output.view(-1,shape1 * shape2)
        output = F.relu(self.dense1(output))
        output = F.relu(self.dense2(output))
        output = self.dense3(output)
        output = F.lsoftmax(output,dim=1)
        return output

    def training_step(self, batch, batch_idx):
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def val_step(self, batch, batch_idx):
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        signals, labels = batch
        outputs = self(signals)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss)
        return loss
    


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 1e-4)
        return optimizer


# COMMAND ----------

def arr_to_dataset(data_arr, label_vec):
  #convert numpy array into tensor dataset
  data_ten = torch.from_numpy(data_arr.transpose(0,2,1)).float()
  label_ten = torch.from_numpy(label_vec.argmax(axis=1))
  #label_ten = torch.from_numpy(label_vec)
  dataset = torch.utils.data.TensorDataset(data_ten, label_ten)
  return dataset

def split_set(features,labels, p=0.7):
  #split data and label array into train, val, test partitions
  n_total = np.shape(features)[0]
  n_train = int(n_total*p)
  n_val = int(n_total * (0.7 + 0.15))
  idx = np.random.permutation(n_total)
  train_mask = idx[:n_train]
  val_mask = idx[n_train:n_val]
  
  test_mask = idx[n_val:]
  trainset = arr_to_dataset(features[train_mask], labels[train_mask])
  valset = arr_to_dataset(features[val_mask], labels[val_mask])
  testset = arr_to_dataset(features[test_mask], labels[test_mask])

  return trainset,valset,testset

# COMMAND ----------

torch.from_numpy(labels)

# COMMAND ----------


def main_training_loop(features,labels,num_tasks, num_proc_per_task):

  """
  
  Main train and test loop
  
  """
  # add imports inside pl_train for pickling to work
  from torch import optim, nn, utils, Tensor
  from torchvision import datasets, transforms
  import pytorch_lightning as pl
  import mlflow
  import os
  
  ############################
  ##### Setting up MLflow ####
  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token
  
  # NCCL P2P can cause issues with incorrect peer settings, so let's turn this off to scale for now
  os.environ['NCCL_P2P_DISABLE'] = '1'
  
  epochs = 5
  batch_size = 1
  dataset = split_set(features,labels)
  # init the autoencoder
  mlflow.pytorch.autolog()
  # Define the tensor specification
  input_tensor_spec = TensorSpec(
    type = np.dtype('float32'),
    shape=(40, 554, 12),# Adjust to the correct data type your model expects
)

  # Create an input schema using the tensor spec
  input_schema = Schema([input_tensor_spec])

  # Define the output schema if necessary; here we assume a basic float output
  # Modify according to your model's actual output
  output_schema = Schema([ColSpec("float","fault")])

  # Create the model signature
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  model =Conv1DFaultModel()
  datamodule = MnistDataModule(dataset,batch_size=1)

  #train the model
  if num_tasks == 1 and num_proc_per_task == 1:
    kwargs = {}
  else:
    kwargs = {"strategy" : "ddp"}
  trainer = pl.Trainer(accelerator='gpu', devices=num_proc_per_task, num_nodes=num_tasks, 
                    limit_train_batches=1000, max_epochs=epochs, **kwargs)
  if run_id is not None:
    mlflow.start_run(run_id=run_id)

  trainer.fit(model=model,datamodule=datamodule)
  trainer.test(model=model,datamodule=datamodule) 
  
  return model, trainer.checkpoint_callback.best_model_path

# COMMAND ----------

#mlflow.end_run()

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import TensorSpec, ColSpec, DataType, Schema
import numpy as np
# Define the tensor specification
input_tensor_spec = TensorSpec(np.dtype('float32'),(-1,12, 554))# Adjust to the correct data type your model expects

# Create an input schema using the tensor spec
input_schema = Schema([input_tensor_spec])

# Define the output schema if necessary; here we assume a basic float output
# Modify according to your model's actual output
output_tensor_spec = TensorSpec(np.dtype('float32'),(-1,8))

output_schema = Schema([output_tensor_spec])

# Create the model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Example usage: Logging a model with MLflow
# Assuming `model` is your trained model object
# with mlflow.start_run():
#     mlflow.pytorch.log_model(model, "model", signature=signature)

print("Input schema for MLflow:")
print(signature)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import mlflow
NUM_TASKS = 1
NUM_PROC_PER_TASK = 1
NUM_PROC = NUM_TASKS * NUM_PROC_PER_TASK



# Start an MLFlow run to automatically track our experiment
with mlflow.start_run() as run:
  run_id= mlflow.active_run().info.run_id
  # TorchDistributor allows us to easily distribute our taining to multiple nodes in a cluster
  (model, ckpt_path) = TorchDistributor(num_processes=NUM_PROC, local_mode=True, use_gpu=USE_GPU).run(main_training_loop,features,labels,NUM_TASKS, NUM_PROC_PER_TASK)
  
  # Log the artifact to MLFlow with a signature that tells consumers what to pass in to the model and what to expect as output
  mlflow.pytorch.log_model(artifact_path="model", 
                          pytorch_model=model,
                          signature=signature,
                          )

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

catalog = "main"
schema = "zach_jacobson"
model_name = "matlab_fault_prediction"

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# Load the model
model_uri = "runs:/294298fd411c45cbbb87ea73e082410a/model"
model = mlflow.pyfunc.load_model(model_uri)
feature = features.astype(np.float32)
# Infer the model signature

# Register the model with the specified signature
mlflow.register_model(model_uri, f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"runs:/294298fd411c45cbbb87ea73e082410a/model")

# COMMAND ----------

data_ten[0]

# COMMAND ----------

features[0].shape
labels[0].shape
data_ten = torch.from_numpy(features.transpose(0,2,1)).float()
label_ten = torch.from_numpy(labels)
label_ten

# COMMAND ----------

feature[0:1].shape

# COMMAND ----------

#feature = features[0].astype(np.float32)
feature = data_ten.numpy()
sample_input = feature[0:1]

model.predict(sample_input)

# COMMAND ----------

# MAGIC %md
# MAGIC # CNN Regression model. Use the distance data frame and refer to this article: https://bamblebam.medium.com/audio-classification-and-regression-using-pytorch-48db77b3a5ec
