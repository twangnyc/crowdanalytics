library(glmnet)
data=read.csv("~/R/project Heart Operation Effect Prediction/data/final/before/completeAN.csv",as.is=TRUE)
data=apply(data,2,as.numeric)
data=as.matrix(data)
row=nrow(data)
col=ncol(data)
Y=as.matrix(data[,col])
X=data[,1:col-1]
X=scale(X)
Pre <- read.csv("~/R/project Heart Operation Effect Prediction/data/final/before/precision.csv")
Name=as.matrix(Pre[,1])
Order=as.matrix(order(abs(Pre[-col,col+1]),decreasing=TRUE))
Order_name=as.matrix(Name[Order])
Sample_number=sample(1:row,960,replace=FALSE)
temp=c(1:row)
Test_number=setdiff(temp,Sample_number)
Order10=Order[1:10,]

Result=matrix(0,nrow=2,ncol=21)
for (i in c(30:50))
{
  lambda=i/100
  Result[,i-29]=parr(lambda)
}