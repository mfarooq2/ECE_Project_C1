import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
#from dask.dataframe import from_pandas
#import dask.dataframe as dd
def data_frame_adder(temp_frame,file_name):
    file_name=file_name.split('.csv')[0]
    temp_frame['file_name']=file_name
    # Splitting the file name and extracting the subject number from it.
    temp_frame['subject']=temp_frame['file_name'].apply(lambda x: int(x.split('subject_')[1].split('_')[0]))
    temp_frame['session']=temp_frame['file_name'].apply(lambda x: int(x.split('subject_')[1].split('_')[1]))
    #temp_frame['type']=temp_frame['file_name'].apply(lambda x: x.split('__')[1])
    temp_frame = temp_frame.drop(columns=['file_name'])
    temp_frame = temp_frame[temp_frame.columns[::-1]]
    temp_frame = temp_frame.sort_values(by=['session','subject'], axis=0)
    col_list1 = list(temp_frame.columns)[:2]
    col_list2 = list(temp_frame.columns)[2:]
    col_list2.sort()
    col_list=col_list1+col_list2

    #temp_frame = swap_columns(temp_frame, 'subject', 'session')
    # Swapping the columns
    temp_frame = temp_frame.reindex(columns=col_list)

    return temp_frame
Training_data_files=glob.glob('TrainingData/*.csv')

file_name_list_x=[]
file_name_list_x_time=[]
file_name_list_y=[]
file_name_list_y_time=[]
for file_name in tqdm(Training_data_files):
    if 'x.csv' in file_name:
        file_name_list_x.append(file_name)
    elif 'x_time.csv' in file_name:
        file_name_list_x_time.append(file_name)
    elif 'y.csv' in file_name:
        file_name_list_y.append(file_name)
    else:
        file_name_list_y_time.append(file_name)
file_name_list_x.sort()
file_name_list_x_time.sort()
file_name_list_y.sort()
file_name_list_y_time.sort()
Subject_list= [f.split('_')[1] for f in file_name_list_x]
Session_list= [f.split('_')[2] for f in file_name_list_x]
x_sensor_dat=pd.DataFrame([])
y_sensor_dat=pd.DataFrame([])
x_time_dat=[]
for sub,sess in zip(Subject_list,Session_list):
    #x_sensor_dat=pd.DataFrame(x_sensor_dat.append(pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x.csv",header=None)))
    
    x_sensor_dat=pd.concat([pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x.csv",header=None),pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x_time.csv",header=None).rename(columns={0:'time_stamp'})],axis=1)

    x_sensor_dat['measurements']=x_sensor_dat.apply(lambda x: [x[0],x[1],x[2],x[3]],axis=1)
    x_sensor_dat=x_sensor_dat[['time_stamp','measurements']]
    x_sensor_dat["time_stamp"]=pd.to_datetime(x_sensor_dat["time_stamp"],unit='s').round('us')
    #x_sensor_dat=x_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    x_sensor_dat['Subject']=int(sub)
    x_sensor_dat['Session']=int(sess)
    #x_sensor_dat=x_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    
    y_sensor_dat=pd.concat([pd.read_csv(f"TrainingData/subject_{sub}_{sess}__y.csv",header=None),pd.read_csv(f"TrainingData/subject_{sub}_{sess}__y_time.csv",header=None).rename(columns={0:'time_stamp'})],axis=1).rename(columns={0:'labels'})
    #y_sensor_dat['labels']=y_sensor_dat.apply(lambda y: [y[0],y[1],y[2],y[3]],axis=1)
    y_sensor_dat=y_sensor_dat[['time_stamp','labels']]
    y_sensor_dat["time_stamp"]=pd.to_datetime(y_sensor_dat["time_stamp"],unit='s').round('ms')
    trp=pd.concat(
        [pd.DataFrame({'labels':0},index=[x_sensor_dat.index[-1]]),pd.DataFrame({'time_stamp':np.max(x_sensor_dat['time_stamp'])},index=[x_sensor_dat.index[-1]])
                         ]
         ,axis=1)
    y_sensor_dat=pd.concat([y_sensor_dat,trp],axis=0)
    trp=pd.concat([pd.DataFrame({'labels':0},index=[x_sensor_dat.index[0]]),pd.DataFrame({'time_stamp':np.min(x_sensor_dat['time_stamp'])},index=[x_sensor_dat.index[0]])],axis=1)
    y_sensor_dat=pd.concat([trp,y_sensor_dat],axis=0)
    # y_sensor_dat=y_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    y_sensor_dat['Subject']=int(sub)
    y_sensor_dat['Session']=int(sess)
    x_sensor_dat=x_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    y_sensor_dat=y_sensor_dat.set_index('time_stamp').asfreq(freq='0.025S', method='bfill')
                                        #.resample('0.025S').interpolate(method='linear')
    x_y_dat=pd.merge(x_sensor_dat,y_sensor_dat, how='inner', left_index=True, right_index=True)
    break