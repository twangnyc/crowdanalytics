fuck_final=function(lambha){  
  #library(glmnet)
  #data=read.csv("~/R/project Heart Operation Effect Prediction/data/final/anzhencomplete.csv",as.is=TRUE)
  #data=apply(data,2,as.numeric)
  #data=as.matrix(data)
  #row=nrow(data)
  #col=ncol(data)
  #Y=as.matrix(data[,col])
  #X=data[,1:col-1]
  #X=scale(X)
  #Pre <- read.csv("~/R/project Heart Operation Effect Prediction/data/final/Precision.csv")
  #Name=as.matrix(Pre[,1])
  #Order=as.matrix(order(abs(Pre[-68,69]),decreasing=TRUE))
  #Order_name=as.matrix(Name[Order])
  Sample_number=sample(1:row,960,replace=FALSE)
  temp=c(1:row)
  Test_number=setdiff(temp,Sample_number)
  Order10=Order[1:10,]
  Sample_X=X[Sample_number,Order10]
  Sample_Y=Y[Sample_number,]
  Sample=cbind(Sample_X,Sample_Y)
  Test_X=X[Test_number,Order10]
  Test_Y=Y[Test_number,] 
  train=glmnet(Sample_X,Sample_Y,family="binomial")
  Test_X=apply(Test_X,2,as.numeric)
  Predict=predict(train,type="response",Test_X,s=0.01)
  for(i in 1:nrow(Predict)){
    if(Predict[i]>lambha){
      Predict[i]=1
    }
    else {
      Predict[i]=0
    }
  }
  Test_Y=as.numeric(as.matrix(Test_Y))
  Temp2=Predict-Test_Y
  count=0;
  positive=0;
  negative=0;
  for(i in 1:nrow(Temp2)){
    if(Temp2[i]==0){}
    else {
      count=count+1
    }
  }
  for(i in 1:nrow(Temp2)){
    if(Temp2[i]==-1){
      negative=negative+1
    }
    else if(Temp2[i]==1){
      positive=positive+1
    }
  }
  #print(count)
  #print(negative)
  return(count/254)
}