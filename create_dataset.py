
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

client = boto3.client('sagemaker',region_name='ap-south-1')
client_runtime = boto3.client('sagemaker-featurestore-runtime',region_name='ap-south-1')


boto_session = boto3.Session(region_name='ap-south-1')
sagemaker_client = boto_session.client(service_name='sagemaker', region_name='ap-south-1')
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name='ap-south-1')

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
athena = boto3.client("athena", region_name='ap-south-1')


#query_string = f'\nSELECT * FROM sagemaker_featurestore.accountfinal-1669202233;'
query_string = f"""
SELECT * FROM "{AccountFinal_table_name}" 
"""
print(query_string)


account_query = AccountFinal_feature_group.athena_query()
account_query.run(query_string=query_string, output_location=f"s3://cloudthat-cs-bucke-new/query_results")
account_query.wait()
dataset = account_query.as_dataframe()


dataset = dataset.fillna(0)
dataset = dataset.drop(['timestamp','segmenttag','reportingmembershortname','write_time','api_invocation_time','is_deleted','paymenthistory1','paymenthistory2'], axis=1)

col_order = ["emiamount"] + list(dataset.drop(["emiamount", "segmenttag-timestamp"], axis=1).columns)


train = dataset.sample(frac=0.80, random_state=0)[col_order]
test = dataset.drop(train.index)[col_order]


train_output_path = pathlib.Path("/opt/ml/processing/output/train")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
train.to_csv(train_output_path / "train.csv", header=True, index=False)
test.to_csv(test_output_path / "test.csv",    header=True, index=False)


import pandas as pd
from sklearn.datasets import dump_svmlight_file
x = train.drop('emiamount', axis=1)
y = train['emiamount']
dump_svmlight_file(X=x, y=y, f='/opt/ml/processing/output/train/train.dat', zero_based=True)

x1 = test.drop('emiamount', axis=1)
y1 = test['emiamount']
dump_svmlight_file(X=x1, y=y1, f='/opt/ml/processing/output/test/test.dat', zero_based=True)
