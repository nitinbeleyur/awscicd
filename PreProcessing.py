
import datetime
import subprocess
import sys
import boto3 
import os 
import time
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas==1.5.1'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'lxml'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'fsspec'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 's3fs'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'sagemaker'])
    from lxml import etree    
except:
    pass

import sagemaker 
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
import pandas as pd 
print("Version",pd.__version__)
ct = datetime.datetime.now()
client = boto3.client('sagemaker',region_name='ap-south-1')
client_runtime = boto3.client('sagemaker-featurestore-runtime',region_name='ap-south-1')

def move_file():
    s3 = boto3.resource('s3')
    copy_source = {
    'Bucket': 'cloudthat-cs-bucket',
    'Key': 'CibilFalse.xml'
                  }
    output_file = str(ct)+' - CibilFalse.xml'

    s3.meta.client.copy(copy_source, 'cloudthat-mlops', output_file)
    s3.Object('cloudthat-cs-bucket', 'CibilFalse.xml').delete()
    return print(output_file)



if __name__ == "__main__":
    
    #input_data_path = os.path.join("", "CibilFalse.xml") 
    input_data_path = os.path.join("/opt/ml/processing/input", "CibilFalse.xml") 
 
    print("Reading input data from {}".format(input_data_path))
    print("Current timestamp - {}".format(ct))
    
    #CreditReport =  pd.read_xml(input_data_path, ".//CreditReport" )
    NameSegment =  pd.read_xml(input_data_path, ".//NameSegment" )
    Header =  pd.read_xml(input_data_path, ".//Header")
    IDSegment =  pd.read_xml(input_data_path, ".//IDSegment" )
    TelephoneSegment =  pd.read_xml(input_data_path, ".//TelephoneSegment" )
    EmailContactSegment =  pd.read_xml(input_data_path, ".//EmailContactSegment" )
    Address =  pd.read_xml(input_data_path, ".//Address" )
    ScoreSegment =  pd.read_xml(input_data_path, ".//ScoreSegment" )
    BureauCharacterstics =  pd.read_xml(input_data_path, ".//BureauCharacterstics" )
    Account =  pd.read_xml(input_data_path, ".//Account" )
    Account_Summary_Segment_Fields =  pd.read_xml(input_data_path, ".//Account_Summary_Segment_Fields" )
    #renaming a column
    Account_Summary_Segment_Fields.rename(columns = {'ReportingMemberShortNameFieldLength':'ReportingMemberShortNameFieldLength-1'}, inplace = True)
    Account_NonSummary_Segment_Fields =  pd.read_xml(input_data_path, ".//Account_NonSummary_Segment_Fields" )
 
    #merging
    AccountFinal = pd.concat([Account, Account_Summary_Segment_Fields, Account_NonSummary_Segment_Fields ], axis=1)
     
        
    
    #Preprocessing NameSegment
    NameSegment['timestamp'] = datetime.datetime.now()
    NameSegment['timestamp'] = NameSegment['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    NameSegment['SegmentTag-timestamp'] = NameSegment['SegmentTag'].astype(str) + NameSegment['timestamp'].astype(str)
    #Preprocessing Header
    Header['timestamp'] = datetime.datetime.now()
    Header['timestamp'] = Header['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    Header['SegmentTag-timestamp'] = Header['SegmentTag'].astype(str) + Header['timestamp'].astype(str)
    #Preprocessing IDSegment
    IDSegment['timestamp'] = datetime.datetime.now()
    IDSegment['timestamp'] = IDSegment['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    IDSegment['SegmentTag-timestamp'] = IDSegment['SegmentTag'].astype(str) + IDSegment['timestamp'].astype(str)
    #Preprocessing TelephoneSegment
    TelephoneSegment['timestamp'] = datetime.datetime.now()
    TelephoneSegment['timestamp'] = TelephoneSegment['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    TelephoneSegment['SegmentTag-timestamp'] = TelephoneSegment['SegmentTag'].astype(str) + TelephoneSegment['timestamp'].astype(str)
    #Preprocessing EmailContactSegment
    EmailContactSegment['timestamp'] = datetime.datetime.now()
    EmailContactSegment['timestamp'] = EmailContactSegment['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    EmailContactSegment['SegmentTag-timestamp'] = EmailContactSegment['SegmentTag'].astype(str) + EmailContactSegment['timestamp'].astype(str)  
    #Preprocessing Addeess
    Address['timestamp'] = datetime.datetime.now()
    Address['timestamp'] = Address['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    Address['SegmentTag-timestamp'] = Address['SegmentTag'].astype(str) + Address['timestamp'].astype(str)
    #Preprocessing ScoreSegment   
    ScoreSegment['timestamp'] = datetime.datetime.now()
    ScoreSegment['timestamp'] = ScoreSegment['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    ScoreSegment['ScoreName-timestamp'] = ScoreSegment['ScoreName'].astype(str) + ScoreSegment['timestamp'].astype(str)
    #Preprocessing BureauCharacterstics 
    BureauCharacterstics['timestamp'] = datetime.datetime.now()
    BureauCharacterstics['timestamp'] = BureauCharacterstics['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    BureauCharacterstics['CV14-timestamp'] = BureauCharacterstics['CV14'].astype(str) + BureauCharacterstics['timestamp'].astype(str)
    #Preprocessing Account_Final    
    AccountFinal['timestamp'] = datetime.datetime.now()
    AccountFinal['timestamp'] = AccountFinal['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    AccountFinal['SegmentTag-timestamp'] = AccountFinal['SegmentTag'].astype(str) + AccountFinal['timestamp'].astype(str)

    
    #Create feature group    
    featuregroups = {
             'Header'        : ['SegmentTag-timestamp','timestamp' ],
             'NameSegment'   : ['SegmentTag-timestamp','timestamp' ],
             'IDSegment'     : ['SegmentTag-timestamp','timestamp' ],
          'TelephoneSegment' : ['SegmentTag-timestamp','timestamp' ],
        'EmailContactSegment': ['SegmentTag-timestamp','timestamp' ],
             'Address'       : ['SegmentTag-timestamp','timestamp' ],
             'ScoreSegment'  : ['ScoreName-timestamp' ,'timestamp' ],
       'BureauCharacterstics': ['CV14-timestamp'      ,'timestamp' ],
             'AccountFinal'  : ['SegmentTag-timestamp','timestamp' ],
                    }
    
 
    for feature_group_name, feature_definitions in featuregroups.items():
        #print(feature_group_name)

        try:
            try:
                time.sleep(2)
                response = client.describe_feature_group(FeatureGroupName='{}'.format(feature_group_name))
                print("Describing - ",feature_group_name," - ", response) 
                print('\n')
            except:
                time.sleep(3)
                response = client.describe_feature_group(FeatureGroupName='{}'.format(feature_group_name))
                print("Describing - ",feature_group_name," - ", response) 
                print('\n')
                
        except:
            time.sleep(3)
            response = client.create_feature_group(
            FeatureGroupName='{}'.format(feature_group_name),
            RecordIdentifierFeatureName='{}'.format(feature_definitions[0]),
            EventTimeFeatureName='{}'.format(feature_definitions[1]),
                FeatureDefinitions=[
              {
                    'FeatureName': '{}'.format(feature_definitions[0]),
                    'FeatureType': 'String'
                },
              {
                    'FeatureName': '{}'.format(feature_definitions[1]),
                    'FeatureType': 'String'
                }
            ],
            OnlineStoreConfig=  {'EnableOnlineStore': True},
            OfflineStoreConfig={
            'S3StorageConfig': {
                'S3Uri': 's3://sagemaker-us-east-2-971709774307/OfflineDataSrore/{}'.format(feature_group_name),
            },
            'DisableGlueTableCreation': False,
            },
            RoleArn =  'arn:aws:iam::971709774307:role/service-role/AmazonSageMaker-ExecutionRole-20210304T113112',
            Description='{}'.format(feature_group_name),
            )
            print("Creating - ",feature_group_name," - ", response) 
            print('\n')
   

    for feature_group_name, feature_name in featuregroups.items():
        #exec("b = {}\nprint('b:', b)".format(feature_group_name))
        exec("b = {}\n".format(feature_group_name))
        dataframe = b.copy()
        #print(dataframe)
        
        datatypes = dataframe.dtypes
        datatypes = dict(datatypes)
        map_datatype = {k: ('Integral' if v == 'int64' else  'Fractional' if v == 'float64' else 'String') for (k, v) in datatypes.items()}
 
        new_feature = dataframe.columns.values.tolist()
        print('CSV Features- ', feature_group_name   ,new_feature)
        print('\n')

        response = client.describe_feature_group(FeatureGroupName='{}'.format(feature_group_name))
        ls = [response['FeatureDefinitions'][i] for i in range(len(response['FeatureDefinitions']))]
        existing_feature =  [i['FeatureName'] for i in ls]
        print('Features Group- ', feature_group_name ,existing_feature)
        print('\n')

        add_feature = [i for i in new_feature if i not in existing_feature]
        print('Feature needs to add in -', feature_group_name ,add_feature)
        print('\n')

        for i in add_feature:
            try:
                response = client.update_feature_group(FeatureGroupName='{}'.format(feature_group_name),
                FeatureAdditions=[
                    {
                        'FeatureName': '{}'.format(i),
                        'FeatureType': '{}'.format(map_datatype[i])
                    },] )
                time.sleep(3)
            except:
                time.sleep(3)
                response = client.update_feature_group(FeatureGroupName='{}'.format(feature_group_name),
                FeatureAdditions=[
                    {
                        'FeatureName': '{}'.format(i),
                        'FeatureType': '{}'.format(map_datatype[i])
                    },] )
                print('Feature Updating --------------------------',response)
                time.sleep(4)
                
                
        response = client.describe_feature_group(FeatureGroupName='{}'.format(feature_group_name))
        ls = [response['FeatureDefinitions'][i] for i in range(len(response['FeatureDefinitions']))]
        existing_feature =  [i['FeatureName'] for i in ls]
        print('Existing Features Now- ',existing_feature)
        print('\n')

        
        
        boto_session = boto3.Session(region_name='ap-south-1')
        sagemaker_client = boto_session.client(service_name='sagemaker', region_name='ap-south-1')
        featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name='ap-south-1')

        feature_store_session = Session(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            sagemaker_featurestore_runtime_client=featurestore_runtime
        )
        '''
        #create session with feature group 
        try:
            try:
                sagemaker_session = sagemaker.Session()
            except:
                time.sleep(5)
                sagemaker_session = sagemaker.Session()
        except Exception as error:
            print(error)
            raise
        '''
        try:
            try:
                featuregroup = FeatureGroup(name='{}'.format(feature_group_name), sagemaker_session=feature_store_session)
                value = featuregroup.ingest(data_frame=dataframe, max_workers=3, wait=True)
                print('Ingestion Result First -----------------------------',value)
            except:
                time.sleep(5)
                featuregroup = FeatureGroup(name='{}'.format(feature_group_name), sagemaker_session=feature_store_session)
                value = featuregroup.ingest(data_frame=dataframe, max_workers=3, wait=True)
                print('Ingestion Result Second-----------------------------',value)
                

        except:
            print('Ingestion failed', feature_group_name ,feature_group_name)

        print('Ingesting - ',feature_group_name,'into', feature_group_name )
        print('\n')
        time.sleep(5)
        print('---------------------',feature_group_name  ,'--------------------------------')
        print('\n')

    
    '''
    Header_output_path = os.path.join("/opt/ml/processing/Header", "Header.csv")
    NameSegment_output_path = os.path.join("/opt/ml/processing/NameSegment", "NameSegment.csv")
    IDSegment_output_path = os.path.join("/opt/ml/processing/IDSegment", "IDSegment.csv")
    TelephoneSegment_output_path = os.path.join("/opt/ml/processing/TelephoneSegment", "TelephoneSegment.csv")
    EmailContactSegment_output_path = os.path.join("/opt/ml/processing/EmailContactSegment", "EmailContactSegment.csv")
    Address_output_path = os.path.join("/opt/ml/processing/Address", "Address.csv")
    ScoreSegment_output_path = os.path.join("/opt/ml/processing/ScoreSegment", "ScoreSegment.csv")
    BureauCharacterstics_output_path = os.path.join("/opt/ml/processing/BureauCharacterstics", "BureauCharacterstics.csv")
    Account_Final_output_path = os.path.join("/opt/ml/processing/Account_Final", 'Account_Final.csv')
    
    pd.DataFrame(Header).to_csv(Header_output_path, header=True, index=False)    
    pd.DataFrame(NameSegment).to_csv(NameSegment_output_path, header=True, index=False)
    pd.DataFrame(IDSegment).to_csv(IDSegment_output_path, header=True, index=False)
    pd.DataFrame(TelephoneSegment).to_csv(TelephoneSegment_output_path, header=True, index=False)
    pd.DataFrame(EmailContactSegment).to_csv(EmailContactSegment_output_path, header=True, index=False)
    pd.DataFrame(Address).to_csv(Address_output_path, header=True, index=False)
    pd.DataFrame(ScoreSegment).to_csv(ScoreSegment_output_path, header=True, index=False)    
    pd.DataFrame(BureauCharacterstics).to_csv(BureauCharacterstics_output_path, header=True, index=False)
    pd.DataFrame(Account_Final).to_csv(Account_Final_output_path, header=True, index=False)
    '''
    #print(move_file())
