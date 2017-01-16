# # # #####################################################
library(s.PP)
library(s.FS)
library(s.ME)
library(s.FE)
source('~/R/my project/mojing contest/code/read_raw_datas.R', echo=F)
##########################################################
alldata<-define_types(raw_master,as.character)
testdata<-define_types(raw_test_master,as.character)
##trans idx to string 
alldata[,1]<-paste('s',alldata[,1],sep='')
testdata[,1]<-paste('s',testdata[,1],sep='')

##########################################################
testdata[,'target']<-NA
all.y<-alldata[,'target']
names(all.y)<-alldata[,'Idx']
set.seed(1314)
vali_num<-sample(1:length(alldata[,1]),10000,replace = F)
vali.key<-alldata[vali_num,'Idx']
train.key<-alldata[-vali_num,'Idx']
test.key<-testdata[,'Idx']
vali.y<-all.y[vali_num]
train.y<-all.y[-vali_num]
alldata[vali_num,'target']<-NA
alldata<-rbind(alldata,testdata)
##########################################################
#drop those strange values
alldata<-stran_sign_handle(alldata,c('','-1'))
########################################################
###relevant feature engineering
source('~/R/my project/mojing contest/code improvement/ci_uudf_deri.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_numvars_deri.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_numvars_cluster_deri.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_add_mac_info.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_equal_deri.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_mk_deriv.R', echo=F)
source('~/R/my project/mojing contest/code improvement/ci_numvars_std_deri.R', echo=F)

#######################################################
temp<-define_types_by_predoc(alldata,predoc)
alldata<-temp$df
num.vars<-temp$undifine[-c(1,(1:37)*12,1+(1:37)*12,446:483,550:557)]
cat.vars<-c(temp$undifine[c((1:37)*12,1+(1:37)*12,446:478,480:483,550:557)],temp$mvc[-1])
useless<-c(temp$undifine[479])
date.var<-"ListingInfo"
cat2bino<-"UserInfo_24"
########################################################
useful<-names(alldata)
useful<-useful[!useful%in%useless]
alldata<-alldata[,useful]
alldata<-define_types(alldata,as.numeric,num.vars)
alldata<-define_types(alldata,as.factor,cat.vars)
alldata[,date.var]<-as.numeric(as.Date(alldata[,date.var]))
alldata[,cat2bino]<-as.numeric(alldata[,cat2bino]=='D')%>%as.character()%>%as.factor()
############################################################
alldata<-data.frame(alldata[,-1],row.names = alldata[,1])
alldata<-miss_handle(alldata,-9999)
alldata<-one_hot_transform(alldata,'target','-9999',0.2)
alldata<-value2interchanged_odds(alldata,'target',4,-9999)
traindata<-alldata[train.key,]
validata<-alldata[vali.key,]
testdata<-alldata[test.key,]
############################################################
useful<-rpart_filter(traindata,'target', controls = rpart.control(minsplit = 10, maxdepth = 15,
                                                                  cp = 0, maxcompete = 1, maxsurrogate = 1, usesurrogate = 1, xval = 4,
                                                                  surrogatestyle = 0))

#######################################################
m.v<--9999
##################################################################
train.mat <-do.call(cbind,traindata[,useful])
train.l<-train.y[train.key]%>%as.character()%>%as.numeric()
dtrain <- xgb.DMatrix(data =train.mat,label = train.l,missing = m.v)

vali.mat <- do.call(cbind,validata[,useful])
vali.l <- vali.y[vali.key]%>%as.character()%>%as.numeric()
dvali <- xgb.DMatrix(data =vali.mat,label = vali.l,missing = m.v)

test.mat <- do.call(cbind,testdata[,useful])
dtest <- xgb.DMatrix(data =test.mat,missing = m.v)
#############################################################
best.para<-list(max.depth =3,scale_pos_weight=0.3,lambda=10,
                gamma=0.1,colsample_bytree=0.3,min_child_weight=10,
                subsample=0.7,eta = 0.1,objective="binary:logistic",
                nthread=4,eval_metric='auc')
# ###########
# best.para<-list(max.depth =8,scale_pos_weight=1,lambda=1,
#                 gamma=0.01,colsample_bytree=0.9,min_child_weight=10,
#                 subsample=0.5,eta = 0.1,objective="binary:logistic",
#                 nthread=4,eval_metric='auc')
set.seed(123)
bsts <- xgb.cv(params=best.para, data=dtrain,nround =10000,verbose = T,nfold=4)
# # ###################################
set.seed(123)
bst <- xgb.train(params=best.para, data=dtrain,nround =613, verbose = 1,watchlist=list('train'= dtrain,'validation'=dvali))
#################################
# probs<-predict(bst,newdata=dtest)
# outcomes<-data.frame(test.key,probs,stringsAsFactors = F)
# names(outcomes)<-c('Idx','score')
# write.csv(outcomes,file='C:/Users/uij/Documents/R/my project/mojing contest/doc/final0.782.csv', row.names = F, fileEncoding = "utf-8")

