temp=matrix(0.5,nrow=1,ncol=1000)
a=matrix(fuck_final(temp[1:1000]),nrow=1,ncol=ncol(temp))
b=matrix(1/ncol(temp),nrow=ncol(temp),ncol=1)
answer=a%*%b
print(answer)
a=t(a)
variance=var(a)