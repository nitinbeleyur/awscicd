
import boto3
import sagemaker
from datetime import datetime
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
sagemaker_session = sagemaker.session.Session()

region = boto3.session.Session().region_name
role = get_execution_role()

#-----------------------------------------------

today = datetime.now()
dic = {}
dic_test ={}
s3 = boto3.resource('s3')
files = s3.meta.client.list_objects(Bucket='cloudthat-cs-bucke-new', Prefix="ProcessingPipeline/")

for i in range(len(files['Contents'])):
    Key = files['Contents'][i]['Key']
    LastModified = files['Contents'][i]['LastModified']
    if Key.rsplit('/', 1)[-1].endswith("train.dat"):
        dt = LastModified.replace(tzinfo=None)
        dic[Key] = (today - dt).total_seconds() 
    elif Key.rsplit('/', 1)[-1].endswith("test.dat"):
        dt_test = LastModified.replace(tzinfo=None)
        dic_test[Key] = (today - dt_test).total_seconds() 

        
value = sorted(dic.items(), key=lambda x: x[1])
value_test = sorted(dic_test.items(), key=lambda x: x[1])
st = "s3://cloudthat-cs-bucke-new/"+"{}".format(value[0][0])
st_test = "s3://cloudthat-cs-bucke-new/"+"{}".format(value_test[0][0])

#----------------------------------------------
model_path = f"s3://cloudthat-cs-bucke-new/PipelineTraining/Model"

bucket = "cloudthat-cs-bucke-new"

from sagemaker.estimator import Estimator


image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region='us-east-2',
    version="1.5-1",
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

    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    verbosity=1,
    objective="reg:linear",
    num_round=50,
)


from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


train_step = TrainingStep(
    name="XgboostTrain",
    estimator=xgb_train,
            inputs={
            "train": st   
            },
       )


#-----------------------------------------------

model = sagemaker.model.Model(
    name="XGBoost-Model",
    image_uri=train_step.properties.AlgorithmSpecification.TrainingImage,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=sagemaker_session,
    role=role,
)

inputs = sagemaker.inputs.CreateModelInput(instance_type="ml.m4.xlarge")

create_model_step = CreateModelStep(name="ModelPreDeployment", model=model, inputs=inputs)


#-----------------------------------------------

register_step = RegisterModel(
    name="XgboostRegisterModel",
    estimator=xgb_train,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="ModelPackageGroupName",
    depends_on=[create_model_step.name]
)


#-----------------------------------------------
from sagemaker.workflow.pipeline import Pipeline

processing_instance_count = 1

pipeline_name = f"TrainingPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    steps=[train_step, create_model_step,register_step ],
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()
