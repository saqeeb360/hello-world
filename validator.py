def mart_train():
    import pandas as pd
#	import numpy as np
    train = pd.read_csv("Train.csv")
    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("LF","Low Fat")
    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("low fat","Low Fat")
    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("reg","Regular")

    train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())
    train['Outlet_Size'] = train['Outlet_Size'].fillna(method="ffill")
    features = train.iloc[:,[0,1,2,3,4,5,6,8,9,10]].values
    labels = train.iloc[:,-1].values
#label Encoding
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    obj_le = []

    for col in [0,2,4,6,7,8,9]:
	    le = LabelEncoder()
	    features[:,col] = le.fit_transform(features[:,col])
	    obj_le.append(le)

    import pickle

    with open("le_objects","wb") as f:
	    pickle.dump(obj_le,f)


#one hot encoding

    obj_ohe=[]
    o1 = OneHotEncoder(categorical_features=[0,4,6,7,8,9])
    features = o1.fit_transform(features).toarray()
    obj_ohe.append(o1)


    with open("objects_ohe","wb") as f_ohe:
	    pickle.dump(obj_ohe,f_ohe)


    from sklearn.model_selection import train_test_split
    features_train,features_test,labels_train,labels_test  = train_test_split(features,labels,test_size=0.10,random_state=0)

    obj_xgboost=[]

    
#	from xgboost import XGBRegressor
#	regressor = XGBRegressor(n_estimators=1000,learning_rate=0.05)
#	regressor.fit(features_train, labels_train,early_stopping_rounds=5,eval_set=[(features_test,labels_test)],verbose=False)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=50,random_state=0)
    regressor.fit(features_train, labels_train)

    obj_xgboost.append(regressor)


    with open("objects_xgboost","wb") as f_xgboost:
	    pickle.dump(obj_xgboost,f_xgboost)


def mart(ii, oi):
    import pickle
    import pandas as pd
    import numpy as np
    
    
    train = pd.read_csv("Train.csv")
    test_data = pd.read_csv("test.csv")
    unique1 = list(train["Outlet_Identifier"].unique())
    print(ii,oi)


    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("LF","Low Fat")
    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("low fat","Low Fat")
    train["Item_Fat_Content"] = train["Item_Fat_Content"].replace("reg","Regular")
    
    train["Item_Weight"] = train["Item_Weight"].fillna(train["Item_Weight"].mean())
    train['Outlet_Size'] = train['Outlet_Size'].fillna(method="ffill")
    
    test_data["Item_Fat_Content"] = test_data["Item_Fat_Content"].replace("LF","Low Fat")
    test_data["Item_Fat_Content"] = test_data["Item_Fat_Content"].replace("low fat","Low Fat")
    test_data["Item_Fat_Content"] = test_data["Item_Fat_Content"].replace("reg","Regular")
    
    test_data["Item_Weight"] = test_data["Item_Weight"].fillna(test_data["Item_Weight"].mean())
    test_data['Outlet_Size'] = test_data['Outlet_Size'].fillna(method="ffill")
    del train["Item_Outlet_Sales"]
    
    df = pd.concat((train,test_data))
    
    data_to_pred = df[(df["Item_Identifier"] == ii)&(df["Outlet_Identifier"] == oi)]
    del data_to_pred["Outlet_Establishment_Year"]


	#label Encoding
#    test = np.array([data_to_pred])
#    with open("new.txt","wb") as n:
#        n.writelines(test)
    with open("le_objects","rb") as f:
        objs =pickle.load(f)

    for col, obj in zip([0,2,4,6,7,8,9], objs):
        data_to_pred.iloc[:,col] = obj.transform(data_to_pred.iloc[:,col])


	#One hot encoding
    with open("objects_ohe","rb") as f_ohe:
        objs_ohe = pickle.load(f_ohe)
        
    data_to_pred = objs_ohe[0].transform(data_to_pred).toarray()

	#Regression model predict
    with open("objects_xgboost","rb") as f_xgboost:
        objects_xgboost = pickle.load(f_xgboost)


    result = objects_xgboost[0].predict(data_to_pred)
    return result