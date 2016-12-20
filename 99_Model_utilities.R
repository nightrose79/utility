############################################################################################################
# Exports submission file and final test dataset
getModelPerformance <- function(data, actual, predicted, bins) {
  
  data$aa <- data[, actual]
  data$pp <- data[, predicted]
    
  #   auc <- getAUCSummary(data, predicted='score_final', actual='actual')
  #   cat('\nAUC = ', auc, sep='') 
  
#   gini <- getGini(data$aa, data$pp)
#   cat('\nGini = ', gini, sep='') 
  
  ngini <- getNormalizedGini(data$aa, data$pp)
  cat('\nNorm. Gini = ', ngini, sep='') 
  
  RMSE <- getRMSE(data$aa, data$pp)
  cat('\nRMSE = ', RMSE, sep='') 
  
  MAE <- getMAE(data$aa, data$pp)
  cat('\nMAE = ', MAE, sep='') 
  
  plotLiftChart(data, response='aa', predicted='pp', numBins=bins, cap=FALSE, cap_pct=0)
}


############################################################################################################
# Calculates the Gini evaluation metric
getGini <- function(a, p) {
  if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
  temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
  temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
  population.delta <- 1 / length(a)
  total.losses <- sum(a)
  null.losses <- rep(population.delta, length(a)) 
  accum.losses <- temp.df$actual / total.losses
  gini.sum <- cumsum(accum.losses - null.losses)
  sum(gini.sum) / length(a)
}

getNormalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

############################################################################################################
# Calculates the RMSE (root mean squared error) evaluation metric
getRMSE <- function(actual, predicted) {
  
  RMSE <- sqrt(mean((actual - predicted)^2))
  
  return(RMSE)
}

############################################################################################################
# Calculates the MAE (Mean Absolute Error) evaluation metric
getMAE <- function(actual, predicted) {
  
  MAE <- mean(abs(actual - predicted))
  
  return(MAE)
}


############################################################################################################
# Calculates the AUC evaluation metric
getAUCSummary <- function(data, predicted, actual, output=TRUE) {
  yhat <- data[, predicted]
  y <- data[, actual]
  
  rocObj <- roc(y, yhat, na.rm=TRUE)
  
  if (output) {
    plot(rocObj, print.auc=TRUE, print.thres=FALSE, reuse.auc=TRUE, col="red")
  }
  
  return(rocObj$auc) 
}


############################################################################################################
# Plot Actuals vs. Predicted scatter
plotActualvPredicted <- function(data, actual, predicted, title='None', xlab='X-label', ylab='Y-label', xlim=NULL, ylim=NULL) {
  data$aa <- data[, actual]
  data$pp <- data[, predicted]
  
  ggplot(data, aes(x=aa, y=pp)) +
    geom_point(alpha=0.33) +
    geom_abline(intercept=0, slope=1, colour = '#0072B2') + 
    coord_fixed() +
    ggtitle(title) +
    xlab(xlab) +
    ylab(ylab) +
    scale_x_continuous(limits=xlim, labels=comma) +
    scale_y_continuous(limits=ylim, labels=comma) +
    theme(plot.title = element_text(size=14)) +
    theme(axis.title.x = element_text(size=12)) +
    theme(axis.title.y = element_text(size=12)) +
    theme(axis.text.x = element_text(size=10, angle=90)) +
    theme(axis.text.y = element_text(size=10))
}

############################################################################################################
# Plot Residual plots
plotActualvResiduals <- function(data, actual, predicted, title='None', xlab='X-label', ylab='Residual (Predicted - Actual)', xlim=NULL, ylim=NULL) {
  data$aa <- data[, actual]
  data$rr <- data[, predicted] - data[, actual] 
  
  ggplot(data, aes(x=aa, y=rr)) +
    geom_point(alpha=0.33) +
    geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
    ggtitle(title) +
    xlab(xlab) +
    ylab(ylab) +
    scale_x_continuous(limits=xlim, labels=comma) +
    scale_y_continuous(limits=ylim, labels=comma) +
    theme(plot.title = element_text(size=14)) +
    theme(axis.title.x = element_text(size=12)) +
    theme(axis.title.y = element_text(size=12)) +
    theme(axis.text.x = element_text(size=10, angle=90)) +
    theme(axis.text.y = element_text(size=10))
}

############################################################################################################
# Plot Residuals By Regressors
plotResidualsByRegressors <- function(data, actual, predicted, regressors, title='None', xlab='X-label', ylab='Residual (Predicted - Actual)', xlim=NULL, ylim=NULL) {
  data$rr <- data[, predicted] - data[, actual] 
  
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    ggplot(data, aes(x=ii, y=rr)) +
      geom_point(alpha=0.33) +
      geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }

  else if (is.character(data$ii) | is.factor(data$ii)) {
     ggplot(data, aes(x=ii, y=rr)) +
        geom_point(alpha=0.33) +
        geom_abline(intercept=0, slope=0, colour = '#0072B2') + 
        ggtitle(title) +
        xlab(xlab) +
        ylab(ylab) +
        scale_y_continuous(limits=ylim, labels=comma) +
        theme(plot.title = element_text(size=14)) +
        theme(axis.title.x = element_text(size=12)) +
        theme(axis.title.y = element_text(size=12)) +
        theme(axis.text.x = element_text(size=10, angle=90)) +
        theme(axis.text.y = element_text(size=10))    
  }

}


############################################################################################################
# Plot Actual vs. Predictors By Regressors
plotActualvPredictedByRegressors <- function(data, actual, predicted, regressors, bins=NULL, title='None', xlab='X-label', ylab='Total Claims', xlim=NULL, ylim=NULL) {
  
  data$aa <- data[ , actual]
  data$pp <- data[ , predicted]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    summary_df1 <- data %>% group_by(ii_bin) %>% summarise(total_loss=sum(aa)) %>% mutate(Group='Actual') %>% ungroup()
    summary_df2 <- data %>% group_by(ii_bin) %>% summarise(total_loss=sum(pp)) %>% mutate(Group='Predicted') %>% ungroup()
    summary_df <- rbind(summary_df1, summary_df2)

    ggplot(summary_df, aes(x=ii_bin, y=total_loss, fill=Group)) +
      geom_bar(position='dodge', stat='identity') +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(breaks=1:bins, limits=xlim) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    summary_df1 <- data %>% group_by(ii) %>% summarise(total_loss=sum(aa)) %>% mutate(Group='Actual') %>% ungroup()
    summary_df2 <- data %>% group_by(ii) %>% summarise(total_loss=sum(pp)) %>% mutate(Group='Predicted') %>% ungroup()
    summary_df <- rbind(summary_df1, summary_df2)
    
    ggplot(summary_df, aes(x=ii, y=total_loss, fill=Group)) +
      geom_bar(position='dodge', stat='identity') +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


############################################################################################################
# Plot % Difference By Regressors
plotPctDiffByRegressors <- function(data, actual, predicted, regressors, bins=NULL, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$aa <- data[ , actual]
  data$pp <- data[ , predicted]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    summary_df <- data %>% group_by(ii_bin) %>% summarise(total_actual_loss=sum(aa), total_predicted_loss=sum(pp), pct_diff=(total_predicted_loss - total_actual_loss)/total_actual_loss) %>% mutate(Group='Actual') %>% ungroup()
    summary_df$pct_diff_dec <- round(summary_df$pct_diff, 2)
    
    ggplot(summary_df, aes(x=ii_bin, y=pct_diff)) +
      geom_bar(stat='identity', fill='#0072B2') +
      geom_text(aes(x=ii_bin, label=pct_diff_dec, vjust=-0.5), size=3) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(breaks=1:bins, limits=xlim) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    summary_df <- data %>% group_by(ii) %>% summarise(total_actual_loss=sum(aa), total_predicted_loss=sum(pp), pct_diff=(total_predicted_loss - total_actual_loss)/total_actual_loss) %>% mutate(Group='Actual') %>% ungroup()
    summary_df$pct_diff_dec <- round(summary_df$pct_diff, 2)
    
    ggplot(summary_df, aes(x=ii, y=pct_diff)) +
      geom_bar(stat='identity', fill='#0072B2') +
      geom_text(aes(x=ii, label=pct_diff_dec, vjust=-0.5), size=3) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}



############################################################################################################
# Plot Boxplots By Regressors
plotClaimsBoxplotsByRegressors <- function(data, claims, regressors, bins=NULL, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$cc <- data[ , claims]
  data$ii <- data[ , regressors]
  
  if (is.numeric(data$ii)) {
    
    data$ii_bin <- binVariable(data$ii, bins=bins)
    data$ii_bin <- as.factor(data$ii_bin)
    
    ggplot(data, aes(x=ii_bin, y=cc)) +
      geom_boxplot() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10)) +
      theme(axis.text.y = element_text(size=10))
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
    
    ggplot(data, aes(x=ii, y=cc)) +
      geom_boxplot() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_y_continuous(limits=ylim, labels=comma) +
      scale_fill_brewer(palette="Set1") +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}


############################################################################################################
# Plot Boxplots By Regressors
plotScatter <- function(data, x, y, title='None', xlab='X-label', ylab='% Difference', xlim=NULL, ylim=NULL) {
  
  data$xx <- data[ , x]
  data$yy <- data[ , y]
  
  if (is.numeric(data$xx)) {
    
    ggplot(data, aes(x=xx, y=yy)) +
      geom_point(alpha=0.33) +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
    
    
  }
  
  else if (is.character(data$ii) | is.factor(data$ii)) {
   
    ggplot(data, aes(yy)) +
      geom_point(alpha=0.33, position = "jitter") +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      scale_x_continuous(limits=xlim, labels=comma) +
      scale_y_continuous(limits=ylim, labels=comma) +
      theme(plot.title = element_text(size=14)) +
      theme(axis.title.x = element_text(size=12)) +
      theme(axis.title.y = element_text(size=12)) +
      theme(axis.text.x = element_text(size=10, angle=90)) +
      theme(axis.text.y = element_text(size=10))
  }
  
}

############################################################################################################
# Plots lift charts
plotLiftChart <- function(data, response, predicted, numBins, cap=TRUE, cap_pct=0) {
  # Requires dplyr
  data$response <- data[, response]  
  data$predicted <- data[, predicted]
  
  # Cap score if cap=TRUE
  if (cap) { 
    data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
  }
  
  # Bin the data into 10 bins
  data$scoreBin <- ntile(data$predicted, numBins)
  #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
  
  # Calculate predicted and realized probability data
  predicted <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = mean(predicted),
              description=' Predicted    '
    )
  
  actual <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = mean(response),
              description=' Actual    '
    )
  
  liftChart <<- rbind(predicted, actual)
  
  # Graph lift chart
  ggplot(liftChart, aes(x=scoreBin, y=stat, fill=description)) +
    geom_bar(position='dodge', stat='identity') +
    scale_fill_brewer(palette="Set1") +
    ggtitle('Lift Chart') +
    xlab('Predicted score bins') +
    ylab('Response') +
    scale_x_continuous(breaks=1:numBins, limits=c(0, numBins+1)) +
    scale_y_continuous(labels=comma) +
    theme_bw() +
    theme(plot.title = element_text(size=20)) +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18)) +
    theme(axis.text.x = element_text(size=14)) +
    theme(axis.text.y = element_text(size=14)) +
    theme(legend.title = element_blank()) +
    theme(legend.text = element_text(size=18)) +
    theme(legend.position='bottom', legend.box='horizontal')
}


############################################################################################################

 ############################################################################################################
 # Plots lift charts
 plotLfChart <- function(data, response, predicted, numBins=10, cap=TRUE, cap_pct=0) {
   # Requires dplyr
   data$response <- data[, response]  
   data$predicted <- data[, predicted]
   
   # Cap score if cap=TRUE
   if (cap) { 
     data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
   }
   
   m <- mean(data$response)
   # Bin the data into 10 bins
   data$scoreBin <- ntile(data$predicted, numBins)
   #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
   
   # Calculate predicted and realized probability data
#    predicted <- data %>%
#      group_by(scoreBin) %>%
#      summarise(stat = mean(predicted),
#                description=' Predicted    '
#      )
   
   actual <- data %>%
     group_by(scoreBin) %>%
     summarise(stat = mean(response),
               description=' Actual Lift'
     ) %>%
     mutate(stat.per=stat/m)
   
   #liftChart <<- rbind(predicted, actual)
    print(actual)

   # Graph lift chart
   ggplot(actual, aes(x=scoreBin, y=stat.per)) +
     #geom_bar(position='dodge', stat='identity') +
     geom_line() + geom_point() + geom_abline(intercept=1,slope=0,colour='red') +
     #scale_fill_brewer(palette="Set1") +
     ggtitle('Lift Chart\n') +
     xlab('\nPredicted score bins\n') +
     ylab('Lift Percentage\n') +
     scale_x_continuous(breaks=1:numBins, limits=c(0, numBins+1)) +
     scale_y_continuous(labels=percent) +
     theme_bw() +
     theme(plot.title = element_text(size=20)) +
     theme(axis.title.x = element_text(size=18)) +
     theme(axis.title.y = element_text(size=18)) +
     theme(axis.text.x = element_text(size=14)) +
     theme(axis.text.y = element_text(size=14)) 

 }
 
 
 ############################################################################################################ 
 # Plots Cumulative Lift Chart
 
 plotCumLiftChart <- function(data, response, predicted, numBins=10, cap=TRUE, cap_pct=0) {
   # Requires dplyr
   data$response <- data[, response]  
   data$predicted <- data[, predicted]
   
   # Cap score if cap=TRUE
   if (cap) { 
     data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
   }
   
   m <- mean(data$response)
   # Bin the data into 10 bins
   data$scoreBin <- ntile(data$predicted, numBins)
   #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
   
   # Calculate predicted and realized probability data
   #    predicted <- data %>%
   #      group_by(scoreBin) %>%
   #      summarise(stat = mean(predicted),
   #                description=' Predicted    '
   #      )
   
   actual <- data %>%
     group_by(scoreBin) %>%
     summarise(stat = sum(response),
               count=n(),
               description=' Actual Lift'
     ) %>%
     arrange(desc(scoreBin)) %>%
     mutate(stat.per=(cumsum(stat)/cumsum(count))/m)
   
   #liftChart <<- rbind(predicted, actual)
   print(actual)
   
   # Graph lift chart
   ggplot(actual, aes(x=scoreBin, y=stat.per)) +
     #geom_bar(position='dodge', stat='identity') +
     geom_line() + geom_point() + geom_abline(intercept=1,slope=0,colour='red') +
     #scale_fill_brewer(palette="Set1") +
     ggtitle('Cumulative Lift Chart\n') +
     xlab('\nPredicted score bins\n') +
     ylab('Cumulative Lift Percentage\n') +
     scale_x_continuous(breaks=1:numBins, limits=c(0, numBins+1)) +
     scale_y_continuous(labels=percent) +
     theme_bw() +
     theme(plot.title = element_text(size=20)) +
     theme(axis.title.x = element_text(size=18)) +
     theme(axis.title.y = element_text(size=18)) +
     theme(axis.text.x = element_text(size=14)) +
     theme(axis.text.y = element_text(size=14)) 
   
 }
 

############################################################################################################ 
# Plots Gini Chart

plotGiniChart <- function(data, response, predicted, numBins=10, cap=TRUE, cap_pct=0) {
  # Requires dplyr
  data$response <- data[, response]  
  data$predicted <- data[, predicted]
  
  # Cap score if cap=TRUE
  if (cap) { 
    data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
  }
  
  s <- sum(data$response)
  n <- length(data$response)
  # Bin the data into 10 bins
  data$scoreBin <- ntile(data$predicted, numBins)
  #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
  gini <- getGini(data$response, data$predicted)

  
  act <- data.frame(scoreBin=0, stat=0, count=0, description='Actual Gains', stat.per=0, cum.count=0, s.per=0)
  actual <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = sum(response),
              count=n(),
              description='Actual Gains'
    ) %>%
    mutate(stat.per=(cumsum(stat)/s), cum.count=(cumsum(count)/n), s.per=stat.per/cum.count)
  
 
  actual$scoreBin <- actual$scoreBin*0.1
  actual <- rbind(act,actual)


  data$scoreBin <- ntile(data$response, 100)
# Calculate predicted and realized probability data
  pr <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = sum(response),
              count=n(),
              description='Actual Gains'
    ) %>%
    mutate(stat.per=(cumsum(stat)/s), cum.count=(cumsum(count)/n), s.per=stat.per/cum.count)
  pr <- rbind(act,pr)
  pr$scoreBin <- pr$scoreBin*0.01
  #liftChart <<- rbind(predicted, actual)
  #print(actual)
  #print(s)

  #print(pr)
  # Graph gains chart
  ggplot(actual, aes(x=scoreBin, y=stat.per)) +
    #geom_bar(position='dodge', stat='identity') +
    geom_line(data=pr, color="red")+
    geom_polygon() + geom_abline(intercept=0,slope=1,color="red") +
    #scale_fill_brewer(palette="Set1") +
    ggtitle('Gini Chart\n') +
    xlab('\nCumulative % of Policies ordered by Predicted\n') +
    ylab('Cumulative % of Losses\n') +
    scale_x_continuous(breaks=seq(0,1,by=0.1), labels=percent) +
    scale_y_continuous(breaks=seq(0,1,by=0.2), labels=percent) +
    theme_bw() +
    theme(plot.title = element_text(size=20)) +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18)) +
    theme(axis.text.x = element_text(size=14)) +
    theme(axis.text.y = element_text(size=14)) 
   
}

############################################################################################################ 
# Plots Cumulative Gains Chart

plotCumGainsChart <- function(data, response, predicted, numBins=10, cap=TRUE, cap_pct=0) {
  # Requires dplyr
  data$response <- data[, response]  
  data$predicted <- data[, predicted]
  
  # Cap score if cap=TRUE
  if (cap) { 
    data$predicted[data$predicted > quantile(data$predicted, 1-cap_pct)] = quantile(data$predicted, 1-cap_pct);  # Cap predicted loss ratios
  }
  
  s <- sum(data$response)
  n <- length(data$response)
  # Bin the data into 10 bins
  data$scoreBin <- ntile(data$predicted, numBins)
  #data$scoreBin = as.numeric(cut2(data$score_final, g=numBins))
  gini <- getGini(data$response, data$predicted)
  
  act <- data.frame(scoreBin=0, stat=0, count=0, description='Actual Gains', stat.per=0, cum.count=0, s.per=0)
  actual <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = sum(response),
              count=n(),
              description='Actual Gains'
    ) %>%
    mutate(stat.per=(cumsum(stat)/s), cum.count=(cumsum(count)/n), s.per=stat.per/cum.count)
  
  actual$scoreBin <- actual$scoreBin*0.1
  actual <- rbind(act,actual)
  #liftChart <<- rbind(predicted, actual)
  #print(actual)
  #print(s)
  # Graph gains chart
  data$scoreBin <- ntile(data$response, 100)
  pr <- data %>%
    group_by(scoreBin) %>%
    summarise(stat = sum(response),
              count=n(),
              description='Actual Gains'
    ) %>%
    mutate(stat.per=(cumsum(stat)/s), cum.count=(cumsum(count)/n), s.per=stat.per/cum.count)
  pr <- rbind(act,pr)
  pr$scoreBin <- pr$scoreBin*0.01
  
  ggplot(actual, aes(x=scoreBin, y=s.per)) +
    #geom_bar(position='dodge', stat='identity') +
    geom_line(data=pr, color="red") +
    geom_polygon() + geom_abline(intercept=0,slope=1,color="red") +
    #scale_fill_brewer(palette="Set1") +
    ggtitle('Cumulative Gains Chart\n') +
    xlab('\nCumulative % of Policies ordered by Predicted\n') +
    ylab('Cumulative % of XS Losses\n') +
    scale_x_continuous(breaks=seq(0,1,by=0.1), labels=percent) +
    scale_y_continuous(breaks=seq(0,1,by=0.2), labels=percent) +
    theme_bw() +
    theme(plot.title = element_text(size=20)) +
    theme(axis.title.x = element_text(size=18)) +
    theme(axis.title.y = element_text(size=18)) +
    theme(axis.text.x = element_text(size=14)) +
    theme(axis.text.y = element_text(size=14)) 
  #+
  # geom_text(data = TextFrame,aes(x = X, y = Y, label = LAB), size = 4)
  #print(pr)
  #print(actual)
}
 
 
# Returns common descriptive stats for datasets
getSummaryStats <- function(data, varList, sortMissing=FALSE, export=FALSE) {
  
  # Gives you percentiles, mean, max, std.dev for list of vectors
  # Required packages: dplyr
  len <- length(varList)
  stats = data.frame(variable = vector(length=len),
                     data_type = vector(length=len),
                     N = vector(length=len),
                     missing = vector(length=len),
                     missingPct = vector(length=len),
                     uniqueVals = vector(length=len),
                     min = vector(length=len),
                     pct01 = vector(length=len),
                     pct02 = vector(length=len),
                     pct05 = vector(length=len),
                     pct25 = vector(length=len),
                     pct50 = vector(length=len),
                     pct75 = vector(length=len),
                     pct95 = vector(length=len),
                     pct98 = vector(length=len),
                     pct99 = vector(length=len),
                     max = vector(length=len),
                     mean = vector(length=len),
                     std.dev = vector(length=len))
  
  for (i in 1:len) {
    var = data[, varList[i]]
    stats$variable[i] = varList[i] 
    stats$data_type[i] = class(var) 
    
    if (is.numeric(var)) {
      stats$N[i] = length(which(!is.na(var)))
      stats$missing[i] = length(which(is.na(var)))
      stats$missingPct[i] = length(which(is.na(var))) / (length(which(!is.na(var))) + length(which(is.na(var))))
      stats$uniqueVals[i] = length(unique(var))
      stats$min[i] = min(var, na.rm=TRUE)
      stats$pct01[i] = quantile(var, 0.01, na.rm=TRUE)
      stats$pct02[i] = quantile(var, 0.02, na.rm=TRUE)
      stats$pct05[i] = quantile(var, 0.05, na.rm=TRUE)
      stats$pct25[i] = quantile(var, 0.25, na.rm=TRUE)
      stats$pct50[i] = quantile(var, 0.50, na.rm=TRUE)
      stats$pct75[i] = quantile(var, 0.75, na.rm=TRUE)
      stats$pct95[i] = quantile(var, 0.95, na.rm=TRUE)
      stats$pct98[i] = quantile(var, 0.99, na.rm=TRUE)
      stats$pct99[i] = quantile(var, 0.99, na.rm=TRUE)
      stats$max[i] = max(var, na.rm=TRUE)
      stats$mean[i] = mean(var, na.rm=TRUE)
      stats$std.dev[i] = sd(var, na.rm=TRUE) 
    }
    
    if (!is.numeric(var)) {
      stats$N[i] = length(which(!is.na(var)))
      stats$missing[i] = length(which(is.na(var)))
      stats$missingPct[i] = length(which(is.na(var))) / (length(which(!is.na(var))) + length(which(is.na(var))))
      stats$uniqueVals[i] = length(unique(var))
      stats$min[i] = NA
      stats$pct01[i] = NA
      stats$pct02[i] = NA
      stats$pct05[i] = NA
      stats$pct25[i] = NA
      stats$pct50[i] = NA
      stats$pct75[i] = NA
      stats$pct95[i] = NA
      stats$pct98[i] = NA
      stats$pct99[i] = NA
      stats$max[i] = NA
      stats$mean[i] = NA
      stats$std.dev[i] = NA
    }
  }
  
  # If you want to sort variables by what's missing (dplyr package needed)
  if (sortMissing == TRUE) {
    stats = arrange(stats, missing)
  }
  
  # To write to csv
  if (export == TRUE) {
    filename <- paste('data/summStats_', deparse(substitute(data)), '.csv',sep='')
    write.csv(stats, file=filename)
  }
  
  View(stats)
  return(stats)
}


############################################################################################################
# Finds frequency counts for factors (characters and integers with less than 20 unique values)
############################################################################################################
getFactorFreqs <- function(data, varList, yvar, export=FALSE) {
  # requires dplyr
  dataName <- deparse(substitute(data))
  len <- length(varList)
  
  if (yvar == 'none') {
    data$y <- NA  
  } else {
    data$y <- data[, yvar]
  }
  
  stats <- NULL
  
  for (i in 1:len) {  
    data$var <- data[, varList[i]]
    
    if (is.factor(data$var) | length(unique(data$var)) <= 20) {
      temp <- data %>%
        group_by(var) %>%
        summarise(
          freq = n(),
          YRate = mean(y, na.rm=TRUE)
        ) %>% ungroup()
      
      temp <- arrange(temp, desc(freq))
      rows <- nrow(temp)
      
      stats_add <- data.frame(variable = vector(length=rows),                        
                              level = vector(length=rows),
                              count = vector(length=rows),
                              percent = vector(length=rows),
                              YRate = vector(length=rows))
      
      stats_add$variable <- varList[i]
      stats_add$level <- temp$var
      stats_add$count <- temp$freq
      stats_add$percent <- temp$freq / sum(stats_add$count, na.rm=TRUE)
      stats_add$YRate <- temp$YRate
      
      stats <- rbind(stats, stats_add)
    }
  }
  
  # To write to csv
  if (export == TRUE) {
    filename <<- paste('data/factorFreqs_', dataName, '.csv',sep='')
    write.csv(stats, file=filename)
  }
  
  View(stats)
  return(stats)
}


############################################################################################################
# Get count and avg responses for factor variables (a.ka. Leave one-out experience variables)
getOneWayVars <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
  # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
  # Requires dplyr
  
  len <- length(varList)
  rowNumCheck.train <- nrow(train)
  rowNumCheck.test <- nrow(test)
  
  train$responseVar <- train[, yvar]
  total_avg_response <- mean(train$responseVar, na.rm=TRUE)  # Fixed only for this contest
  
  for (i in 1:len) {
    train$groupingVar <- train[, varList[i]]
    test$groupingVar <- test[, varList[i]]   
    
    df <- train %>%
      group_by(groupingVar) %>%
      summarise(
        freq = n() - 1,
        YRate = mean(responseVar, na.rm=TRUE)
      ) %>% ungroup()
    
    train <- left_join(train, df, by='groupingVar')
    
    train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate')])
    test <- left_join(test, train_tmp, by='groupingVar')
    names(test)[which(names(test)=='freq')] <- 'dummyFreq'
    names(test)[which(names(test)=='YRate')] <- 'dummyRate'
    test$dummyFreq <- test$dummyFreq + 1
    test$dummyFreq[is.na(test$dummyFreq)] <- 0
    
    ids <- which(is.na(test$dummyRate))
    test$dummyRate[ids] <- total_avg_response
    test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
    
    if (freq) {
      names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
    } else {
      id <- which(names(test)=='dummyFreq')
      test[, id] <- NULL
    }
    
    names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    # Leave one out adjustment for train data
    train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
    train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
    train$YRate[train$freq == 0] <- total_avg_response
    set.seed(10)
    train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
    
    if (freq) {
      names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
    } else {
      id <- which(names(train)=='freq')
      train[, id] <- NULL
    }
    
    names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    train$groupingVar <- NULL;
    test$groupingVar <- NULL;
  }
  
  train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
  
  if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
  
  if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
  
  test <<- test
  return(train)
}


getOneWayVars_retTest <- function(train, test, varList, yvar, freq=TRUE, cred=0, rand=0) {
  # freq=TRUE when you want the factor counts; set cred > 0 for credibility adjustment; rand > 0 for random shocking
  # Requires dplyr
  
  len <- length(varList)
  rowNumCheck.train <- nrow(train)
  rowNumCheck.test <- nrow(test)
  
  train$responseVar <- train[, yvar]
  total_avg_response <- mean(train$responseVar, na.rm=TRUE)  # Fixed only for this contest
  
  for (i in 1:len) {
    train$groupingVar <- train[, varList[i]]
    test$groupingVar <- test[, varList[i]]   
    
    df <- train %>%
      group_by(groupingVar) %>%
      summarise(
        freq = n() - 1,
        YRate = mean(responseVar, na.rm=TRUE)
      ) %>% ungroup()
    
    train <- left_join(train, df, by='groupingVar')
    
    train_tmp <- unique(train[, c('groupingVar', 'freq', 'YRate')])
    test <- left_join(test, train_tmp, by='groupingVar')
    names(test)[which(names(test)=='freq')] <- 'dummyFreq'
    names(test)[which(names(test)=='YRate')] <- 'dummyRate'
    test$dummyFreq <- test$dummyFreq + 1
    test$dummyFreq[is.na(test$dummyFreq)] <- 0
    
    ids <- which(is.na(test$dummyRate))
    test$dummyRate[ids] <- total_avg_response
    test$dummyRate[-ids] <- (test$dummyRate[-ids] + (total_avg_response * cred / test$dummyFreq[-ids])) * (test$dummyFreq[-ids] / (test$dummyFreq[-ids] + cred))
    
    if (freq) {
      names(test)[which(names(test)=='dummyFreq')] <- paste(varList[i], '_freq', sep='')  
    } else {
      id <- which(names(test)=='dummyFreq')
      test[, id] <- NULL
    }
    
    names(test)[which(names(test)=='dummyRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    # Leave one out adjustment for train data
    train$YRate <- (train$YRate - (train$responseVar / (train$freq+1))) * (train$freq+1)/(train$freq)
    train$YRate <- (train$YRate + (total_avg_response * cred / train$freq)) * (train$freq / (train$freq + cred))
    train$YRate[train$freq == 0] <- total_avg_response
    set.seed(10)
    train$YRate <- train$YRate * (1+(runif(nrow(train))-0.5) * rand)
    
    if (freq) {
      names(train)[which(names(train)=='freq')] <- paste(varList[i], '_freq', sep='')
    } else {
      id <- which(names(train)=='freq')
      train[, id] <- NULL
    }
    
    names(train)[which(names(train)=='YRate')] <- paste(varList[i], '_', yvar, 'Rate', sep='')
    
    train$groupingVar <- NULL;
    test$groupingVar <- NULL;
  }
  
  train$responseVar <- NULL; train$groupingVar <- NULL; test$groupingVar <- NULL;
  
  if(nrow(train) != rowNumCheck.train) print('Error: Different number of rows in train data. Bad join!')
  
  if(nrow(test) != rowNumCheck.test) print('Error: Different number of rows in test data. Bad join!')
  
  return(test)
}



############################################################################################################
# Changes class type of variables
makeFactor <- function(data, varList) {
  for (i in varList) {
    data[, i] <- as.factor(data[, i])
  }
  
  return(data)
}



############################################################################################################
# Puts multiple ggplots on same page
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols)) 
  }
  
  if (numPlots==1) {
    print(plots[[1]])  
  } else {
    # Set up the page
    grid.newpage() 
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,                              
                                      layout.pos.col = matchidx$col))     
    }  
  }
}


Evaluate<-function(actual,predicted){
  print(paste0("Normalized Gini: ", getNormalizedGini(actual,predicted)))
  print(paste0("R-squared: ", caret::R2(predicted,actual)))
  print(paste0("Mae: ", mae(actual,predicted)))
  print(paste0("Min of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(0))))
  print(paste0("5th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.05))))
  print(paste0("10th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.1))))
  print(paste0("25th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.25))))
  print(paste0("Median of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.5))))
  print(paste0("75th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.75))))
  print(paste0("90th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.9))))
  print(paste0("95th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.95))))
  print(paste0("98th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.98))))
  print(paste0("99th Percentile of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(.99))))
  print(paste0("Max of abs. Residuals: ", quantile(abs(actual-predicted),probs=c(1))))
  
  
  par(mfrow=c(1,2))
  plot(actual,predicted)
  abline(0,1,col='red')
  plot(predicted,actual-predicted)
  abline(0,0,col='red')
  
}


############################################################################################################
percentile <- function(x) rank(x, na.last='keep')/length(which(!is.na(x)))

############################################################################################################
binVariable <- function(x, bins) {
  # requires dplyr package
  ntile(x, bins)
}


############################################################################################################
imputeWithMean <- function(data, vars) {
  data <- as.data.frame(data)
  
  for (i in vars) {
    data[is.na(data[, i]), i] <- mean(as.numeric(data[, i]), na.rm=TRUE)
  }
    
  return(data)
}


############################################################################################################
standardize <- function(x) (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)


############################################################################################################
mScale <- function(array) {
  x<-(array-min(array))/(max(array)-min(array))
  return(x)
}

mScaleTest<-function(array,min,max){
  x<-(array-min)/(max-min)
  return(x)
}

