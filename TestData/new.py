import pandas as pd
x_sensor_dat=pd.DataFrame([])
y_sensor_dat=pd.DataFrame([])
x_time_dat=[]
x_sensor_dat=pd.DataFrame([])
y_sensor_dat=pd.DataFrame([])
x_time_dat=[]
for sub,sess in zip(Subject_list,Session_list):
    #x_sensor_dat=pd.DataFrame(x_sensor_dat.append(pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x.csv",header=None)))
    
    x_sensor_dat=pd.concat([pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x.csv",header=None),pd.read_csv(f"TrainingData/subject_{sub}_{sess}__x_time.csv",header=None).rename(columns={0:'time_stamp'})],axis=1)

    x_sensor_dat['measurements']=x_sensor_dat.apply(lambda x: [x[0],x[1],x[2],x[3]],axis=1)
    x_sensor_dat=x_sensor_dat[['time_stamp','measurements']]
    x_sensor_dat["time_stamp"]=pd.to_datetime(x_sensor_dat["time_stamp"],unit='s').round('us')
    x_sensor_dat=x_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    x_sensor_dat['Subject']=sub
    x_sensor_dat['Session']=sess
    
    
    y_sensor_dat=pd.concat([pd.read_csv(f"TrainingData/subject_{sub}_{sess}__y.csv",header=None),pd.read_csv(f"TrainingData/subject_{sub}_{sess}__y_time.csv",header=None).rename(columns={0:'time_stamp'})],axis=1).rename(columns={0:'labels'})
    #y_sensor_dat['labels']=y_sensor_dat.apply(lambda y: [y[0],y[1],y[2],y[3]],axis=1)
    y_sensor_dat=y_sensor_dat[['time_stamp','labels']]
    y_sensor_dat["time_stamp"]=pd.to_datetime(y_sensor_dat["time_stamp"],unit='s').round('ms')
    trp=pd.concat([pd.DataFrame({'Subject':sub},index=[x_sensor_dat.index[-1]]),pd.DataFrame({'Session':sess},index=[x_sensor_dat.index[-1]]),pd.DataFrame({'labels':np.nan},index=[x_sensor_dat.index[-1]])],axis=1)
    y_sensor_dat=pd.concat([y_sensor_dat,trp],axis=0)
    
    trp=pd.concat([pd.DataFrame({'Subject':sub},index=[x_sensor_dat.index[0]]),pd.DataFrame({'Session':sess},index=[x_sensor_dat.index[0]]),pd.DataFrame({'labels':np.nan},index=[x_sensor_dat.index[0]])],axis=1)
    y_sensor_dat=pd.concat([y_sensor_dat,trp],axis=0)
    y_sensor_dat=y_sensor_dat.set_index('time_stamp').asfreq('0.025S')
    y_sensor_dat['Subject']=sub
    y_sensor_dat['Session']=sess
    
    break
#sensor_data_total=pd.DataFrame({'x_sensor_dat':x_sensor_dat,'x_time_dat':_time_dat})        