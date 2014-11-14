AN<- read.csv("~/R/project Heart Operation Effect Prediction/Data_20130201/after process/full_AN.csv", header=F)
rownumber=nrow(AN)
columnnumber=ncol(AN)
indicator=matrix(NA,rownumber,columnnumber)
for (i in c(1:rownumber))
{
  for (j in c(1:columnnumber))
  {
    if (is.na(AN[i,j]))
{
      indicator[i,j]=0
    }
    else
    {
      indicator[i,j]=1
    }
  }
}
Correlation=cor(AN,AN,"pairwise.complete.obs")
Precision=solve(Correlation)

