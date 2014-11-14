fuck=function(pppp,k){
  RAN=read.csv(pppp)
  rownumber=nrow(RAN)
  columnnumber=ncol(RAN)
  indicator=matrix(NA,rownumber,columnnumber)
  for (i in c(1:rownumber))
  {
    for (j in c(1:columnnumber))
    {
      if (is.na(RAN[i,j]))
      {
        indicator[i,j]=0
      }
      else
      {
        indicator[i,j]=1
      }
    }
  }
  #Correlation=cor(RAN,RAN,"pairwise.complete.obs")
  Precision=read.csv("~/R/project Heart Operation Effect Prediction/Precision.csv")
  
  
  #minimal位?玫暮???
  minn<-function(n) function(x) order(x, decreasing = FALSE)[n]#??一?????业???n小??元?氐?????
  
  
  ##RAN预????
  AN<-scale(RAN)
  #write.csv(AN[,29],file="24standard.csv")
  ##Precision matrix ??一??
  
  partialrho<-matrix(0,nrow(Precision),ncol(Precision))
  for(i in c(1:nrow(Precision)))
  {
    for (j in c(1:ncol(Precision)))
    {
      partialrho[i,j]=Precision[i,j] /sqrt(Precision[i,i]*Precision[j,j])
    }
  }
  
  partialrho<-abs(partialrho)
  
  ##kNN
  
  for(j in c(1:columnnumber))   # j ??attribute
  { 
    
    partialrho[,j]<-partialrho[,j]/sum(partialrho[,j])
    
    for(i in c(1:nrow(AN)))       # ??i ????    
    {
      if(is.na(AN[i,j]))          # ??i ???说?j??attribute缺失??
      { 
        tmp<-which(!is.na(AN[,j])) #??每??attribute??????l???撕偷?i???说木???
        count<-sum(is.na(AN[,j]))
        ttmp<-c(1:ncol(AN))
        ttmp<-ttmp[-j]
        
        
        dvector<-matrix(0,1,nrow(AN))
        dneighbor<-matrix(0,1,k)  #??录?????????募????诰拥木???值
        dindicator<-matrix(0,1,k) #??录?????????募????诰拥??卤?
        
        for(n in tmp)              #??n????
        {        
          for(m in ttmp)           #??m??指??
          {
            
            if(is.na(AN[n,m])&&is.na(AN[i,m]))
            {
              d=2                   # ?诘?m??attribute?希???i???撕偷?n???硕???缺??
            }else if((!is.na(AN[n,m]))&&(!is.na(AN[i,m]))) {
              d=(AN[n,m]-AN[i,m])^2
            }else if(is.na(AN[n,m])&&(!is.na(AN[i,m]))){        # ??m??attribute?希?只?械?i????缺失
              d=1+(AN[i,m])^2
            }else if(is.na(AN[i,m])&&(!is.na(AN[n,m]))){
              d=1+(AN[n,m])^2
            }else{
              d=0
            }
            
            dvector[1,n]=dvector[1,n]+d*partialrho[m,j]  
          }
          
        }
        dvector<-sqrt(dvector)
        for(p in c(1:k))
        {
          dindicator[1,p]<-apply(dvector,1,minn(p+count))
          dneighbor[1,p]<-dvector[1,dindicator[p]]  #????????小?牡??冉洗???
          
        }
        
        AN[i,j]<-sum(AN[dindicator,j])/k   
        
      }
    }
    
  }

flag=0
  if(flag)
  {
    for(j in c(1:columnnumber))   # j 是attribute
    {   
      for(i in c(1:nrow(AN)))       # 第i 个人    
      {
        if(is.na(indicator[i,j]))          # 第i 个人第j个attribute缺失了
        { 
          tmp<-which(!is.na(indicator[,j])) #对每个attribute计算第l个人和第i个人的距离
          count<-sum(is.na(indicator[,j]))
          ttmp<-c(1:ncol(AN))
          ttmp<-ttmp[-j]
          
          
          dvector<-matrix(0,1,nrow(AN))
          dneighbor<-matrix(0,1,k)  #记录距离最近的几个邻居的距离值
          dindicator<-matrix(0,1,k) #记录距离最近的几个邻居的下标
          
          for(n in tmp)              #第n个人
          {        
            for(m in ttmp)           #第m个指标
            {
              d=(AN[n,m]-AN[i,m])^2
            }
            
            dvector[1,n]=dvector[1,n]+d*partialrho[m,j]  
          }
          
        }
        dvector<-sqrt(dvector)
        for(p in c(1:k))
        {
          dindicator[1,p]<-apply(dvector,1,minn(p+count))
          dneighbor[1,p]<-dvector[1,dindicator[p]]  #距离从最小的到比较大的
          
        }
        
        AN[i,j]<-sum(AN[dindicator,j])/k   
        
      }
    }
    
  }

  write.csv(AN,file="completeAN.csv")
  
  
  return(AN)
}
fuck2=cmpfun(fuck)
AN_trial=fuck2("~/R/project Heart Operation Effect Prediction/data/anzhenbefore.csv",10)
