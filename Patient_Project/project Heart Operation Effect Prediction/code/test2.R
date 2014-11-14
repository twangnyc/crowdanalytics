P=matrix(0,nrow(L),ncol(L))
for (i=c(1:nrow))
{
  for (j=c(1:ncol))
  {
    P[i,j]=L[i,j]
  }
}