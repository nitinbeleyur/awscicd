
import sys
import subprocess
import pathlib
import time
import boto3
import pandas as pd

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "s3fs"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec"])
    import s3fs
    import fsspec
except:
    print('Import failed')


import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

client = boto3.client('sagemaker',region_name='us-east-2')
client_runtime = boto3.client('sagemaker-featurestore-runtime',region_name='us-east-2')


boto_session = boto3.Session(region_name='us-east-2')
sagemaker_client = boto_session.client(service_name='sagemaker', region_name='us-east-2')
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name='us-east-2')

feature_store_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

#feature_store_session = sagemaker.Session()

s3_client = boto3.client("s3")



#Feature Groups & Data Tables 
AccountFinal_feature_group = FeatureGroup(name="AccountFinal", sagemaker_session=feature_store_session)
IDSegment_feature_group    = FeatureGroup(name="IDSegment"   , sagemaker_session=feature_store_session)

AccountFinal_table_name = (AccountFinal_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"])
IDSegment_table_name =    (IDSegment_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"])

athena_database_name = AccountFinal_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["Database"]
print(athena_database_name)
print(f'AccountFinal_table_name: {AccountFinal_table_name}')
print(f'IDSegment_table_name: {IDSegment_table_name}')

# query athena table
athena = boto3.client("athena", region_name='us-east-2')


#query_string = f'\nSELECT * FROM sagemaker_featurestore.accountfinal-1669202233;'
query_string = f"""
SELECT * FROM "{AccountFinal_table_name}" 
"""
print(query_string)


account_query = AccountFinal_feature_group.athena_query()
account_query.run(query_string=query_string, output_location=f"s3://sagemaker-cloudthat-mlosp/query_results")
account_query.wait()
dataset = account_query.as_dataframe()


col_order = ["emiamount"] + list(dataset.drop(["emiamount", "segmenttag-timestamp"], axis=1).columns)
train = dataset.sample(frac=0.80, random_state=0)[col_order]
test = dataset.drop(train.index)[col_order]

#test.shape,train.shape


# Write train, test splits to output path
train_output_path = pathlib.Path("/opt/ml/processing/output/train")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
train.to_csv(train_output_path / "train.csv", index=False)
test.to_csv(test_output_path / "test.csv", index=False)
