from sensor.entity import artifact_entity,config_entity
from sensor.exception import SensorException
from sensor.components import data_ingestion
from sensor.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from sensor import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN
import pandas as pd



class DataTransformation:
    
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_tranformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy = "constant",fill_value = 0)
            robust_scaler = RobustScaler()

            pipeline = Pipeline(steps = [
                ('Imputer', simple_imputer),
                ('RobustScaler' , robust_scaler)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)

    def initiaite_data_trasformation(self,)-> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path1)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path1)            

            #selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis =1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis =1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)


            # dropping _id columns from test and train column
            """if "_id" in target_feature_train_arr.columns:
                logging.info(f"dropping column _id")
                target_feature_train_arr.drop("_id",axis=1)
                logging.info(f"row and column in df : {target_feature_train_arr.shape}")
                return target_feature_train_arr"""

            transformation_pipeline = DataTransformation.get_data_tranformer_object()      
            transformation_pipeline.fit(input_feature_train_df) 

            # transformation on input features    
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy = "minority")
            logging.info(f"Before Resampling in Training Set Input : {input_feature_train_arr.shape}  Target : {target_feature_train_arr}")
            input_feature_train_arr,target_feature_train_arr =smt.fit_resample(input_feature_train_arr,target_feature_train_arr)
            logging.info(f"After Resampling in Training Set Input : {input_feature_train_arr.shape}  Target : {target_feature_train_arr}")

            logging.info(f"Before Resampling in Training Set Input : {input_feature_test_arr.shape}  Target : {target_feature_test_arr}")
            input_feature_test_arr,target_feature_test_arr =smt.fit_resample(input_feature_test_arr,target_feature_test_arr)
            logging.info(f"After Resampling in Training Set Input : {input_feature_test_arr.shape}  Target : {target_feature_test_arr}")

            #Target Encoder
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_arr]

            #save numpy array
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transform_train_path,
                                         array = train_arr)
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transform_train_path,
                                         array = test_arr)

            utils.save_object(file_path = self.data_transformation_config.transform_object_path,
                                 obj = transformation_pipeline)

            utils.save_object(file_path = self.data_transformation_config.target_encoder_path,
                                     obj = label_encoder)
            
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transform_train_path = self.data_transformation_config.transform_train_path,
                transform_test_path = self.data_transformation_config.transform_test_path,
                target_encoder_path= self.data_transformation_config.target_encoder_path
                
                )

            logging.info(f"Data Transformation Object")
            return data_transformation_artifact


        except Exception as e:
            raise SensorException(e, sys)


