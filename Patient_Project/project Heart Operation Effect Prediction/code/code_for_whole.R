fucku=function(lamdha){  
  library(glmnet)
  data=read.csv("~/R/project Heart Operation Effect Prediction/Data_20130201/after process/an.csv",as.is=TRUE)
  data[,3]=factor(data[,3])
  data[,20]=factor(data[,20])
  data[,51]=factor(data[,51])
  data=apply(data,2,as.numeric)
  data=as.matrix(data)
  data=knnImputation(data)
  row=nrow(data)
  col=ncol(data)
  Y=as.matrix(data[,col])
  X=data[,1:col-1]
  X=scale(X)
  logistic=glmnet(X,Y,family="binomial")
  Rank=as.matrix(coef(logistic,s=0.001))
  Rank=as.matrix(Rank[-1,])
  Name=as.matrix(row.names(Rank))
  Order=as.matrix(order(abs(Rank),decreasing=TRUE))
  Order_name=as.matrix(Name[Order])
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
  Predict=predict(train,type="response",Test_X,s=0.001)
  for(i in 1:nrow(Predict)){
    if(Predict[i]>lambdha){
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
  print(count)
  print(negative)
  #return(count/254)
}