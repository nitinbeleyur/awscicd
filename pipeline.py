
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
    code="Final-Pre-Processing22-11-22.py",
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

training_job_output_path = f"s3://sagemaker-us-east-2-971709774307/training_jobs"

#train_instance_param = ParameterString(
#    name="TrainingInstance",
#    default_value="ml.m4.xlarge",)


from sagemaker.estimator import Estimator


image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region='us-east-2',
    version="1.0-1",
    py_version="py3",
    instance_type="ml.m5.xlarge"
)
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=model_path,
    role=role,
)
xgb_train.set_hyperparameters(
    objective="reg:linear",
    num_round=50,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    silent=0
)


from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


train_step = TrainingStep(
    name="XgboostTrain",
    estimator=xgb_train,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=create_dataset_step_process.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri
        )
    },
)


from sagemaker.processing import ScriptProcessor


script_eval = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="script-credit-saison",
    role=role,
)

from sagemaker.workflow.properties import PropertyFile


evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)
step_eval = ProcessingStep(
    name="CreditSaisonEval",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=create_dataset_step_process.properties.ProcessingOutputConfig.Outputs[
                "test_data"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
    ],
    code="evaluation.py",
    property_files=[evaluation_report],
)



from sagemaker.model import Model


model = Model(
    image_uri=image_uri,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=role,
)

from sagemaker.inputs import CreateModelInput


inputs = CreateModelInput(
    instance_type="ml.m5.large",
    accelerator_type="ml.eia1.medium",
)

from sagemaker.workflow.steps import CreateModelStep


step_create_model = CreateModelStep(
    name="CreditSaisonCreateModel",
    model=model,
    inputs=inputs,
)


model_package_group_name = f"CreditSaisonModelPackageGroupName"

from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.step_collections import RegisterModel

model_approval_status = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval" )

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"
    )
)

step_register = RegisterModel(
    name="CreditSaisonRegisterModel",
    estimator=xgb_train,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics
)
    
    
#-----------------------------------------------

from sagemaker.workflow.pipeline import Pipeline

processing_instance_count = 1

pipeline_name = f"Pipeline24-11-2022"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_count,model_approval_status],
    steps=[step_process, create_dataset_step_process,train_step , step_eval , step_create_model , step_register],
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()
