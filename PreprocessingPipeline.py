
!pip install --upgrade sagemaker

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost

sagemaker_session = sagemaker.session.Session()

region = boto3.session.Session().region_name
role = get_execution_role()
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
#-----------------------------------------------

sklearn_processor = SKLearnProcessor(framework_version='1.0-1',
                                     role=role,
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1)

input_data = "s3://cloudthat-cs-bucket/CibilFalse.xml"
model_path = f"s3://sagemaker-cloudthat-mlosp/model_path"

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
    

step_process = ProcessingStep(
    name="XML-to-FeatureStore",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
     outputs=[],  
    code="PreProcessing.py",
    processor=sklearn_processor
)

#-----------------------------------------------

create_dataset_processor = SKLearnProcessor(framework_version="1.0-1",
                                            role=role,
                                            instance_type="ml.m5.xlarge",
                                            instance_count=1)



create_dataset_step_process = ProcessingStep(
    name="Create-Dataset",
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train_data", source="/opt/ml/processing/output/train"
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test_data", source="/opt/ml/processing/output/test"
        ),
    ], 
    code="create_dataset.py",
    processor=create_dataset_processor,
    depends_on=[step_process.name]
)


#-----------------------------------------------

from sagemaker.workflow.pipeline import Pipeline

processing_instance_count = 1

pipeline_name = f"ProcessingPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count],
    steps=[step_process, create_dataset_step_process],
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()
