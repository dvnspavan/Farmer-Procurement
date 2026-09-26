"""Train reproducible pipelines; run after generate_dataset.py."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
ROOT=Path(__file__).parent; MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
CAT=['Procurement_Center','Crop_Type','Weather_Condition','Time_Slot']; NUM=['Month','Historical_Farmer_Arrivals','Festival_or_Holiday','Quantity_Quintals','Number_of_Vehicles','Available_Machines','Number_of_Staff','Average_Processing_Time','Slot_Capacity']
def pipe(est): return Pipeline([('pre',ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),CAT),('num',SimpleImputer(strategy='median'),NUM)])),('model',est)])
def reg_metrics(y,p): return {'MAE':round(mean_absolute_error(y,p),2),'RMSE':round(mean_squared_error(y,p)**.5,2),'R2':round(r2_score(y,p),3)}
def main():
 d=pd.read_csv(ROOT/'procurement_data.csv'); X=d[CAT+NUM]; tr,te=train_test_split(range(len(d)),test_size=.2,random_state=42)
 arrival=pipe(RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=1)); arrival.fit(X.iloc[tr],d['Actual_Farmer_Arrivals'].iloc[tr]); ap=arrival.predict(X.iloc[te])
 waiting=pipe(RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=1)); waiting.fit(X.iloc[tr],d['Waiting_Time_Minutes'].iloc[tr]); wp=waiting.predict(X.iloc[te])
 congestion=pipe(RandomForestClassifier(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=1)); congestion.fit(X.iloc[tr],d['Congestion_Level'].iloc[tr]); cp=congestion.predict(X.iloc[te]); pr,re,f,_=precision_recall_fscore_support(d['Congestion_Level'].iloc[te],cp,average='weighted',zero_division=0)
 cluster_cols=['Quantity_Quintals','Waiting_Time_Minutes','Actual_Farmer_Arrivals','Average_Processing_Time']; scaler=StandardScaler(); z=scaler.fit_transform(d[cluster_cols]); km=KMeans(n_clusters=4,n_init=15,random_state=42).fit(z); d['Cluster']=km.labels_; d.to_csv(ROOT/'procurement_data.csv',index=False)
 for name,obj in [('arrival_model.pkl',arrival),('waiting_time_model.pkl',waiting),('congestion_model.pkl',congestion),('kmeans_model.pkl',km),('cluster_scaler.pkl',scaler)]: joblib.dump(obj,MODELS/name)
 metrics={'arrival':reg_metrics(d['Actual_Farmer_Arrivals'].iloc[te],ap),'waiting':reg_metrics(d['Waiting_Time_Minutes'].iloc[te],wp),'congestion':{'accuracy':round(accuracy_score(d['Congestion_Level'].iloc[te],cp),3),'precision':round(pr,3),'recall':round(re,3),'f1':round(f,3),'confusion_matrix':confusion_matrix(d['Congestion_Level'].iloc[te],cp).tolist()}}
 (MODELS/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
