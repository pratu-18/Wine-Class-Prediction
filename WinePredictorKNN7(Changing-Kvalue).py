import pandas as pd
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix
from sklearn.preprocessing import StandardScaler

#all steps are called MLOps pipelines

def MarvellousClassifier(DataPath):
    
    border="-"*40
    
    #------------Step 1: load the dataset from csv file---------------
    
    print(border)
    print("Step 1: load the dataset from csv file")
    print(border)
    
    df=pd.read_csv(DataPath)
    print(border)
    print("Some entries from dataset")
    print(df.head())
    print(border)


    #--------------Step 2 :  clean the dataset-------------------------------
    print(border)
    print("Step 2 :  Clean the dataset")
    print(border)
    
    df.dropna(inplace=True)             #removes all missing values(NaN,None, or NaT) from directly place or  directly excel
    print("Shape of data set :",df.shape)     #178x14
    print("Total records :",df.shape[0])     #178
    print("Total columns :",df.shape[1])     #14
    print(border) 
    
    
    #-----------------Step 3: Separate independent and dependent variables----------------
    print(border)
    print("Step 3: Separate independent and dependent variables")
    print(border)
    
    
    X = df.drop(columns=['Class'])
    Y = df['Class']
    print("Shape of X:",X.shape)
    print("Shape of Y: ",Y.shape)
    print(border)
    print("Input columns :",X.columns.tolist())
    print("Output columns : Class")
    
    
    #-----------------Step 4: Split the dataset for traning and testing--------------------------
    
    print(border)
    print("Step 4: Split the dataset for traning and testing")
    print(border)
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=42,stratify=Y)
    
    print(border)
    print("Details of traning and testing data ")
    print("Shape of  X_train: ",X_train.shape)
    print("Shape of X_test :",X_test.shape)
    print("Shape of  Y_train: ",Y_train.shape)
    print("Shape of Y_test :",Y_test.shape)
    print(border)
    
    
    #-----------------Step 5: Feature scaling (high low value la yeka range mde anr)----------------------------
    print(border)
    print("Step 5: Feature scaling")
    print(border)
    
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)
    print("Feature scaling Done")
    print(border)
    
    
   #--------------Changing value of K in loop format ----------------
   
   #-----------Step 6: Hyperparameter tunning-----------------
    print(border)
    print("Step 6: Hyperparameter tunning")
    print(border)
    
    
    accuracy_scores=[]
    K_values=range(1,21)
    
    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model=model.fit(X_train_scaled,Y_train)
        Y_pred=model.predict(X_test_scaled)
        accuracy=accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)
        
    print("Acuracy report by changing multiple values of K")
    for no in accuracy_scores:
        # print(no)
        print(no*100)
    print(border)
    
    
    #-------step 7: visulisation--------------
    print(border)
    print("Graphical representaion")
    print(border)
    
    plt.figure(figsize=(8,5))
    plt.plot(K_values,accuracy_scores, marker="o")
    plt.title("K values v/s accurancy")
    plt.xlabel("Value of K")
    plt.ylabel("Accurancy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()



def main():
    MarvellousClassifier("WinePredictor.csv")

if __name__=="__main__":
    main()