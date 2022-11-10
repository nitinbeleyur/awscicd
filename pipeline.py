
!pip install --upgrade sagemaker

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor

region = boto3.session.Session().region_name

role = get_execution_role()

from sagemaker.sklearn.processing import SKLearnProcessor

sklearn_processor = SKLearnProcessor(framework_version='1.0-1',
                                     role=role,
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1)


input_data = "s3://cloudthat-cs-bucket/CibilFalse.xml"


from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
    

step_process = ProcessingStep(
    name="XMLParsing-PreProcessing",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
     outputs=[
        
        ProcessingOutput(output_name="Header", source="/opt/ml/processing/Header"),
        ProcessingOutput(output_name="NameSegment", source="/opt/ml/processing/NameSegment"),
        ProcessingOutput(output_name="IDSegment", source="/opt/ml/processing/IDSegment"),        
        ProcessingOutput(output_name="TelephoneSegment", source="/opt/ml/processing/TelephoneSegment"),
        ProcessingOutput(output_name="EmailContactSegment", source="/opt/ml/processing/EmailContactSegment"),
        ProcessingOutput(output_name="Address", source="/opt/ml/processing/Address"),
        ProcessingOutput(output_name="ScoreSegment", source="/opt/ml/processing/ScoreSegment"),
        ProcessingOutput(output_name="BureauCharacterstics", source="/opt/ml/processing/BureauCharacterstics"),
        ProcessingOutput(output_name="Account_Final", source="/opt/ml/processing/Account_Final"),

         
              ],  
    code="preprocessing.py",
    processor=sklearn_processor
)


from sagemaker.workflow.pipeline import Pipeline

processing_instance_count = 1

pipeline_name = f"XMLParsing-PreProcessing-Pipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count],
    steps=[step_process],
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()
