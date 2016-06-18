# Diagnostic of Breast Cancer
setwd("/Users/manel/Documents/Universidad/MIRI/Q1B/MVA/data_sets")
# Reading the dataset 
# Delete all variables
rm(list=ls(all=TRUE))
dbc <- read.csv('WDBC.dat', header = FALSE)

# 0. Basic analysis 
names.cols <- c('ID.number','Diagnosis','M.radius','M.texture','M.permiter','M.area',
                'M.smoothness','M.compctness','M.concavity','M.concave.points',
                'M.symmmetry','M.fractal.dimension','SE.radius','SE.texture','SE.perimeter',
                'SE.area','SE.smoothness','SE.compactness','SE.concavity','SE.concave.points',
                'SE.symmetry','SE.fractal.dimension','W.radius','W.texture','W.perimeter',
                'W.area','W.smoothness','W.compactness','W.concavity','W.concave.points',
                'W.symmetry', 'W.fractal.dimension')
colnames(dbc) <- names.cols
rownames(dbc)<- dbc$ID.number
dbc <- dbc[,-1]
summary(dbc)
dim(dbc)
rownames(dbc)
colnames(dbc)
str(dbc)

# Detecting missing values (just to be sure)
sum(is.na(dbc))
# outlier detection
# Parallel implementation of the Lof algorithm
library(Rlof)
outlier.scores <- lof(dbc[,-1], k=5)
# pick top 10 as outliers
outliers <- order(outlier.scores, decreasing=T)[1:10]
print(outliers)
# Due to our non-expert knowledge in this field, we could not considered
# if these outliers are important points or must be eliminated, so we 
# just will keep them untouched for the analysis  

# Divide the dataset in training (2/3 of the total) and test (1/3)
N <- dim(dbc)[1]
learn <- sample(1:N, round(2/3*N))
dbc.train <- dbc[learn,]
dbc.test <- dbc[-learn,]
dbc <- rbind(dbc.train, dbc.test)
# So as to extract important information, we decide to perform a PCA
library("FactoMineR")
par(mfrow = c(1,2))
dbc.pca <- PCA(dbc, ind.sup =380:569, quali.sup = 1)
par(mfrow = c(1,1))
# plot of the individuals (train and test)
dbc.pca.ind <- rbind(dbc.pca$ind$coord,dbc.pca$ind.sup$coord)
plot(dbc.pca.ind, col = dbc$Diagnosis)
# Decide the significant dimensions so as to analyze the problem
# Plot of the eigenvalues
plot(dbc.pca$eig$eigenvalue,type = "b", main = "Eigenvalues")
dbc.pca$eig
# Kaiser rule: more than 80% of variance explained, so we keep 5 PCs

# Exploratory analysis 
# Formation of PCs
pc.rot <-varimax(dbc.pca$var$coord)
p <- ncol(dbc[,2:31])
Phi.rot <- pc.rot$loadings[1:p,]
lmb.rot <- diag(t(Phi.rot) %*% Phi.rot)
X <- as.matrix(dbc[,-1])
Xs <- scale(X, center = TRUE)
Psi_stan.rot <- Xs %*% solve(cor(X)) %*% Phi.rot
Psi.rot <- Psi_stan.rot %*% diag(sqrt(lmb.rot))
cols <- character(nrow(dbc))
# Plot of the individuals in the rotated PCs
# using Diagnosis as color
cols[dbc$Diagnosis == 'B'] <- "green"
cols[dbc$Diagnosis == 'M'] <- "red"
plot(Psi.rot,type="n",main = "Individuals")
text(Psi.rot,labels=rownames(dbc),col=cols)
abline(h=0,v=0,col="gray")
# Explanation of the PCs using the variables available in the dataset
nd <- 5
for (k in 1:nd)
{
  cat("PC ",k, "\n")
  print(condes(cbind(Psi.rot[,k],dbc),1))
}

# Clustering 
# Coordinates of the individuals in the PCs
# n=nrow(dbc.pca.ind)
# hierarchical clustering
dbc.pca.ind.train <- dbc.pca$ind$coord
d <- dist(dbc.pca.ind.train,method = "euclidean")
hc <- hclust(d, method = "ward.D2")
plot(hc)
barplot(sort(hc$height, decreasing = TRUE)[1:40])
# Looking at the barplot jumps, we conclude that 2 groups 
nc = 2
c1 <- cutree(hc,nc)
# centroids of the clusters
cdg <- aggregate(as.data.frame(dbc.pca.ind.train), list(c1),mean)
# Summary of the number of individuals assigned to a cluster
k2 <- kmeans(dbc.pca.ind.train, centers = cdg[,2:(nd+1)], iter.max = 100)
k2.cluster <- k2$cluster
table(k2.cluster)
# Interpret and name the obtained clusters and represent them
# in the first factorial display
par(mfrow = c(1,1))
plot(dbc.pca.ind.train,type = "n", main = "Clustered Individuals")
text(dbc.pca.ind.train,labels=rownames(dbc.pca.ind.train),col=k2.cluster)
catdes(cbind(as.factor(k2.cluster),dbc.train),num.var = 1)
# Assing test data to the corresponding cluster 
dbc.pca.test <- dbc.pca$ind.sup$coord
k2.centers <- k2$centers
k2.centers[1,]
dbc.pca.test[1,]
dim(dbc.pca.test)
test.centers <- 1:190
dbc.pca.test <- as.matrix(dbc.pca.test)

for (i in 1:dim(dbc.pca.test)[1]) {
  # print(dist(rbind(dbc.pca.test[i,],k2.centers[1,])))
  # print(dist(rbind(dbc.pca.test[i,],k2.centers[2,])))
    if ( dist(rbind(dbc.pca.test[i,],k2.centers[1,])) >= dist(rbind(dbc.pca.test[i,], k2.centers[2,]))) {
      test.centers[i] <- 'red'
    }
    else {
        test.centers[i] <- 'black'
    }
}
table(test.centers)
# Plot the test individuals with the train individuals
plot(dbc.pca.ind.train, col = k2.cluster)
points(dbc.pca.test, col = test.centers, pch = 4)

# Trying to predit Diagnosis type
dbc.pca.ind <- as.data.frame(dbc.pca.ind)
dbc.pca.ind['Diagnosis'] <- dbc$Diagnosis
# Trying to predict the type of breast cancer using random forest
library(TunePareto)
# Prepare a crossvalidation 10x10 method to get the best model with
# several classifiers
k <- 10
CV.folds <- generateCVRuns(dbc.pca.ind[1:379,]$Diagnosis, ntimes=1, nfold=k, stratified=TRUE)
cv.results <- matrix (rep(0,4*k),nrow=k)
colnames (cv.results) <- c("k","fold","TR error","VA error")
dbc.pca.ind.train <- dbc.pca.ind[1:379,]
cv.results[,"TR error"] <- 0
cv.results[,"VA error"] <- 0
cv.results[,"k"] <- k
resp.var <- which(colnames(dbc.pca.ind)=='Diagnosis')

for (j in 1:k) {
  va <- unlist(CV.folds[[1]][[j]])
  tr <- dbc.pca.ind.train[-va,]
rf <- randomForest(formula = Diagnosis ~., data = tr, xtest = dbc.pca.ind.train[va,-resp.var],
                   ytest = dbc.pca.ind.train[va,resp.var])
cv.results[j,"TR error"] <- 1 - sum(diag(rf$confusion[,-11]) / sum(rf$confusion[,-11]))
cv.results[j,"VA error"] <- 1 - sum(diag(rf$test$confusion[,-11]) / sum(rf$test$confusion[,-11]))
cv.results[j,"fold"] <- j
}
cv.results
mean(cv.results[,'VA error'])

library(randomForest)
rf1 <- randomForest(formula = Diagnosis ~., data = dbc.pca.ind, subset = 1:379, importance = TRUE, 
                    xtest = dbc.pca.ind[380:569,-6],ytest = dbc.pca.ind[380:569,6])
print(rf1)
# Random Forest works so well with this dataset

