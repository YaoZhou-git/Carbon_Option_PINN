clear;clc;clf;
if isempty(getenv('CARBON_OPTION_DATA_DIR')), error('Set CARBON_OPTION_DATA_DIR to a licensed legacy-data directory.'); end

A=xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'期货结算价(连续)_欧盟排放配额(EUA).csv'),1,'B1719:B2814');
% Step 1: Generate some random time series data  
  
n = length(A); % 数据点的数量  

  
% Step 2: Create a GARCH(1,1) model  
model = garch('GARCHLags',1,'ARCHLags',1);  
  
% Step 3: Estimate model parameters  
[fit,~,logL] = estimate(model, A);  
  
% Step 4: Compute and plot the conditional variances (volatilities)  
[V,F] = infer(fit, A);  
  
% 绘制原始时间序列和其波动性  
figure;  
subplot(2,1,1);  
plot(A);  
title('Time Series');  
subplot(2,1,2);  
plot(sqrt(V)); % 绘制标准差作为波动性的度量  
title('Volatility');
