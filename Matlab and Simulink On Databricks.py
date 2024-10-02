# Databricks notebook source
# MAGIC %md
# MAGIC Kit (JDK) 1.8 or later
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC Databricks Unified Analytics Platform is a cloud-based service for running your analytics in one place,
# MAGIC from reliable and performant data pipelines to state-of-the-art machine learning. From the original
# MAGIC creators of Apache Spark and MLflow, it provides data science and engineering teams ready-to-use clusters
# MAGIC with optimized Apache Spark and various ML frameworks coupled with powerful collaboration capabilities
# MAGIC to improve productivity across the ML lifecycle.
# MAGIC
# MAGIC With Matlab, you can leverage models and simulations, compile code into python, and deploy as
# MAGIC
# MAGIC ## Features
# MAGIC
# MAGIC * Access data via JDBC
# MAGIC * Interactively access a Databricks cluster using DBConnect
# MAGIC * Access data via DBFS
# MAGIC * Control Databricks clusters and operation via the REST API
# MAGIC * **Deploy compiled MATLAB applications to a Databricks cluster**
# MAGIC * **Deploy compiled MATLAB functions as libraries on a Databricks cluster**
# MAGIC
# MAGIC

# COMMAND ----------

#dbutils.fs.cp("dbfs:/dbfs/Mathworks/dbx.faults-24.1.0-py3-none-any.whl", "/Volumes/main/zach_jacobson/mathworks", recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #When choosing cluster config, 3 sepearte tests were run, generating 10 simulations. 
# MAGIC ##Cluster Config as follows:
# MAGIC - Driver and worker type: r6id.xlarge, 32 GB memory, 4 cores, 4 worker nodes, Photon and autoscaling enabled (Delta Cache Memory Optimized)
# MAGIC - Driver and worker type: r5dn.2xlarge, 64 GB memory, 8 cores, 4 workers, Photon and autoscaling enbaled (Delta Cache Memory Optimized)
# MAGIC - Driver and worker type: r5dn.2xlarge, 128 GB memory, 16 cores, 4 workers, Photon and autoscaling enbaled (Delta Cache Memory Optimized)
# MAGIC
# MAGIC ##The first compute instance above runs for about 3.5 hours, the second instance above runs for ~1 hour and the third instance above runs for ~1 hour 

# COMMAND ----------

# MAGIC %md
# MAGIC #Intro
# MAGIC ### The workflow described in this document programmatically steps through fault sequences on each three-phase bus, two-phase bus and single-phase bus in the IEEE 123 Node Test Feeder. There are 70 three-phase buses, with 7 fault sequences possible per bus, giving a total of 490 faults for the three-phase buses. There are 3 two-phase buses, with 3 fault sequences possible per bus, giving a total of 9 faults for the two-phase buses. There are 55 single-phase buses, with 1 fault sequence possible per bus, giving a total of 55 faults for the single-phases buses. Overall, there are a total of 554 fault sequences in the system. The range in cell 4 provides several fault resistance values that is used in this example. Each simulation is ran for 55.5 seconds and then stops and starts a new one

# COMMAND ----------

# MAGIC %md
# MAGIC ![](/Volumes/main/zach_jacobson/mathworks/image_1.png)

# COMMAND ----------

# Databricks notebook source
# Import special functions from wrapper
from dbx.faults.wrapper import run_top_level_faults_output_names
from dbx.faults.wrapper import run_top_level_faults_output_schema
from dbx.faults.wrapper import run_top_level_faults_mapPartitions
from dbx.faults.wrapper import run_top_level_faults_applyInPandas
from dbx.faults.wrapper import run_top_level_faults_mapInPandas


# COMMAND ----------

# COMMAND ----------
#Resistance (Ron and Rg) of faults as random lograthmic like function of the resistance
#Simulation 1: 10 simulations on 
from pyspark.sql import functions as f
N = 10

R = spark.range(N)

#Create logarithmic inputs for simulink model
DF = R.withColumn("Ron", f.pow(10.0, -1.0 * (N - R['id'].cast('double')) / N)) \
.withColumn("Rg", f.pow(10.0, -2.0 * (N - R['id'].cast('double')) / N)) \
.withColumn("StopTime", f.lit(1.0)) \
.select('Ron', 'Rg', 'StopTime')

DF.display()




# COMMAND ----------

# COMMAND ----------
#V1mag_13: Voltage magnitude 1 phase bus 13
#V1ph_13: Voltage phase angle 1 phase bus 13
import time
OUT2 = DF.groupBy('Ron').applyInPandas(run_top_level_faults_applyInPandas, run_top_level_faults_output_schema).cache()
#OUT2 = DF.groupBy('Ron').applyInPandas(run_top_level_faults_applyInPandas, run_top_level_faults_output_schema).cache()

# COMMAND ----------

#display(OUT2)

# COMMAND ----------

# MAGIC %sh ls -lh /
# MAGIC

# COMMAND ----------

OUT2.write.format("delta").mode("overwrite").saveAsTable("main.zach_jacobson.fault_simulation1")

# COMMAND ----------

#11 steps per simulation. 1000 simulations which is short and produces 11000 inputs (11*1000 = 11000 inputs)
#Outputs all fault types. Location of fault type will be located by the bus number that is listed in the output. 

