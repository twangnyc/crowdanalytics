library("clime")
anzhen= read.csv("~/R/project Heart Operation Effect Prediction/data/final/anzhen.csv")
Cor_matrix=cor(anzhen,anzhen,"pairwise.complete.obs")
Pre=clime(Cor_matrix,lambda=0.00000000001,sigma=TRUE)
Pre_matrix=Pre$Omegalist[[1]]
write.csv(Pre_matrix,file="C:/Users/wuhao/Documents/R/project Heart Operation Effect Prediction/Precision.csv")