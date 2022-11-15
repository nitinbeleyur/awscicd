
import datetime
import boto3 
import os 
import pandas as pd 
from lxml import etree

print("Version",pd.__version__)
ct = datetime.datetime.now()

def move_file():
    s3 = boto3.resource('s3')
    copy_source = {
    'Bucket': 'cloudthat-cs-bucket',
    'Key': 'CibilFalse.xml'
                  }
    output_file = str(ct)+' - CibilFalse.xml'

    s3.meta.client.copy(copy_source, 'cloudthat-mlops', output_file)
    s3.meta.client.copy(copy_source, 'sagemaker-cloudthat-mlosp', output_file)
    s3.Object('cloudthat-cs-bucket', 'CibilFalse.xml').delete()
    return print(output_file)



if __name__ == "__main__":
    
    input_data_path = "s3://cloudthat-cs-bucket/CibilFalse.xml"
    #input_data_path = os.path.join("", "CibilFalse.xml") 
    #input_data_path = os.path.join("/opt/ml/processing/input", "CibilFalse.xml") 
 
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
    Account_Final = pd.concat([Account, Account_Summary_Segment_Fields, Account_NonSummary_Segment_Fields ], axis=1)
     
        
    
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
    Account_Final['timestamp'] = datetime.datetime.now()
    Account_Final['timestamp'] = Account_Final['timestamp'].apply(lambda x: datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    Account_Final['SegmentTag-timestamp'] = Account_Final['SegmentTag'].astype(str) + Account_Final['timestamp'].astype(str)
   
    Header_output_path = os.path.join("s3://cloudthat-cs-bucket/", "Header.csv")
    pd.DataFrame(Header).to_csv(Header_output_path, header=True, index=False)    

    '''
    #CreditReport_output_path = os.path.join("/opt/ml/processing/CreditReport", "CreditReport.csv")
    Header_output_path = os.path.join("/opt/ml/processing/Header", "Header.csv")
    NameSegment_output_path = os.path.join("/opt/ml/processing/NameSegment", "NameSegment.csv")
    IDSegment_output_path = os.path.join("/opt/ml/processing/IDSegment", "IDSegment.csv")
    TelephoneSegment_output_path = os.path.join("/opt/ml/processing/TelephoneSegment", "TelephoneSegment.csv")
    EmailContactSegment_output_path = os.path.join("/opt/ml/processing/EmailContactSegment", "EmailContactSegment.csv")
    Address_output_path = os.path.join("/opt/ml/processing/Address", "Address.csv")
    ScoreSegment_output_path = os.path.join("/opt/ml/processing/ScoreSegment", "ScoreSegment.csv")
    BureauCharacterstics_output_path = os.path.join("/opt/ml/processing/BureauCharacterstics", "BureauCharacterstics.csv")
    #Account_output_path = os.path.join("/opt/ml/processing/Account", "Account.csv")
    #Account_Summary_Segment_Fields_output_path = os.path.join("/opt/ml/processing/Account_Summary_Segment_Fields", "Account_Summary_Segment_Fields.csv")
    #Account_NonSummary_Segment_Fields = os.path.join("/opt/ml/processing/Account_NonSummary_Segment_Fields", "Account_NonSummary_Segment_Fields.csv")
    Account_Final_output_path = os.path.join("/opt/ml/processing/Account_Final", 'Account_Final.csv')

    #pd.DataFrame(CreditReport).to_csv(CreditReport_output_path, header=False, index=False)
    pd.DataFrame(Header).to_csv(Header_output_path, header=True, index=False)    
    pd.DataFrame(NameSegment).to_csv(NameSegment_output_path, header=True, index=False)
    pd.DataFrame(IDSegment).to_csv(IDSegment_output_path, header=True, index=False)
    pd.DataFrame(TelephoneSegment).to_csv(TelephoneSegment_output_path, header=True, index=False)
    pd.DataFrame(EmailContactSegment).to_csv(EmailContactSegment_output_path, header=True, index=False)
    pd.DataFrame(Address).to_csv(Address_output_path, header=True, index=False)
    pd.DataFrame(ScoreSegment).to_csv(ScoreSegment_output_path, header=True, index=False)    
    pd.DataFrame(BureauCharacterstics).to_csv(BureauCharacterstics_output_path, header=True, index=False)
    #pd.DataFrame(Account).to_csv(Account_output_path, header=False, index=False)
    #pd.DataFrame(Account_Summary_Segment_Fields).to_csv(Account_Summary_Segment_Fields_output_path, header=False, index=False)
    #pd.DataFrame(Account_NonSummary_Segment_Fields).to_csv(Account_NonSummary_Segment_Fields, header=False, index=False)
    pd.DataFrame(Account_Final).to_csv(Account_Final_output_path, header=True, index=False)
    '''
    #print(move_file())
