#####################################################
library(s.PP)
library(s.FS)
library(s.ME)
library(s.FE)
########################################################
traindata<-inner_join(rawdata_X, rawdata_y, by ="uid")
traindata<-rename(traindata,def=y)
traindata<-data.frame(traindata[,-1],row.names = traindata[,1])
traindata<-define_types_by_predoc(traindata,features_type)
traindata[,"def"]<-as.character(traindata[,"def"])%>%as.factor()
#############################################################
pred.df<-raw_predict
pred.df<-data.frame(pred.df[,-1],row.names = pred.df[,1])
pred.df<-define_types_by_predoc(pred.df,features_type)
##############################################################
unl.data<-data.frame(raw_unlabeled[,-1],row.names = raw_unlabeled[,1])
unl.data<-define_types_by_predoc(unl.data,features_type)
############################################################
temp<-get_types(traindata,"def")
dis.vars<-temp[[1]]
con.vars<-temp[[2]]
all.vars<-c(dis.vars,con.vars)
############################################################
df<-rbind(traindata[,names(pred.df)],unl.data[,names(pred.df)])
df<-trans2miss(df,-1,'-1')
train.df<-trans2miss(traindata,-1,'-1')
pred.df<-trans2miss(pred.df,-1,'-1')
###############################################
bayes.fit<-bayes_fit(train.df[,c(dis.vars,'def')],pred.df[,dis.vars],'def')
train.df<-bayes_transform(train.df,bayes.fit)
pred.df<-bayes_transform(pred.df,bayes.fit)
#######################################################
train.df<-miss_handle(train.df,-9999)
pred.df<-miss_handle(pred.df,-9999)
m.v<--9999
#######################################################
best.para<-list(max.depth =8,scale_pos_weight=1400/13458,lambda=550,
                gamma=0.1,colsample_bytree=0.4,min_child_weight=3,
                subsample=0.7,eta = 0.02,objective="binary:logistic",
                early.stop.round=100,nthread=7)
bst<-xg_train(train.df,names(pred.df),'def',nround=10000,para=best.para)
probs<-xg_predict(pred.df,names(pred.df),bst)
outcomes<-cbind(row.names(pred.df),probs)
write.csv(outcomes,file='outcome1602034e.csv')
