clear all; clc;
if isempty(getenv('CARBON_OPTION_DATA_DIR')), error('Set CARBON_OPTION_DATA_DIR to a licensed legacy-data directory.'); end

%% 1. 数据读取与预处理
% 假设 A1 为第一履约期期货(如Dec22)，A2 为第二履约期期货(如Dec23)
% 注意：此处假设 A1 和 A2 长度相同且日期对齐
A1_raw = xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'EUA Future Dec24 and Dec25 price in 24.xlsx'), 1, 'B3:B249'); 
A2_raw = xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'EUA Future Dec24 and Dec25 price in 24.xlsx'), 1, 'D3:D249');

pi_penalty = 100; % 罚金参数，需根据实际市场设置
kappa = exp(-0.03 * 1); % 假设折现率3%，期间1年，需根据实际调整

% 提取状态变量 (根据理论推导公式)
% xi2 = norminv(A2/pi)
% xi1 = norminv((A1 - kappa*A2)/pi)
A2_scaled = A2_raw ./ pi_penalty;
A1_diff = (A1_raw - kappa .* A2_raw) ./ pi_penalty;

% 边界修剪：防止 norminv 出现 Inf
A2_scaled(A2_scaled >= 1) = 0.999; A2_scaled(A2_scaled <= 0) = 0.001;
A1_diff(A1_diff >= 1) = 0.999; A1_diff(A1_diff <= 0) = 0.001;

xi2 = norminv(A2_scaled);
xi1 = norminv(A1_diff);

data = [xi1'; xi2']; % 组合成 2 x N 矩阵
timeSteps = 0:(length(A1_raw)-1);
T = length(A1_raw) + 30; % 履约截止日期需大于观测日期

%% 2. 联合参数估计
% 初始猜测: [h, alpha, beta, rho]
initParams = [0, 0.1, 0.1, 0]; 

% 优化选项
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', 'MaxIter', 10000);

% 执行最大似然估计
[params_opt, fval] = fminunc(@(p) -jointLogLikelihood(data, timeSteps, p, T), initParams, options);

h_star = params_opt(1);
alpha_star = params_opt(2);
beta_star = params_opt(3);
rho_star = params_opt(4);

fprintf('估计参数结果:\n h: %.4f\n alpha: %.4f\n beta: %.4f\n rho: %.4f\n', h_star, alpha_star, beta_star, rho_star);

%% 3. 残差计算与诊断
[w1, w2] = jointResiduals(data, timeSteps, params_opt, T);

figure(1);
subplot(2,1,1); plot(w1); title('Residuals \xi_1');
subplot(2,1,2); plot(w2); title('Residuals \xi_2');

figure(2);
subplot(2,2,1); autocorr(w1); title('ACF \xi_1');
subplot(2,2,2); autocorr(w2); title('ACF \xi_2');
subplot(2,2,3); qqplot(w1); title('QQ Plot \xi_1');
subplot(2,2,4); qqplot(w2); title('QQ Plot \xi_2');

%% --- 核心函数库 ---

function logL = jointLogLikelihood(data, timeSteps, params, T)
    h = params(1); alpha = params(2); beta = params(3); rho = params(4);
    
    % 相关系数约束
    if abs(rho) >= 1, logL = -1e10; return; end
    
    logL = 0;
    N = size(data, 2);
    
    for i = 2:N
        dt = timeSteps(i) - timeSteps(i-1);
        t_now = timeSteps(i-1);
        
        % 计算每个周期的 z_t
        z1 = beta * (T - t_now)^(-alpha);
        z2 = beta * (T + 365 - t_now)^(-alpha); % 假设第二周期比第一周期远一年
        
        % 均值向量 mu (2x1)
        mu1 = xi_mean(data(1,i-1), h, z1, dt);
        mu2 = xi_mean(data(2,i-1), h, z2, dt);
        
        % 方差与协方差
        s1_sq = xi_var(z1, dt);
        s2_sq = xi_var(z2, dt);
        % 跨期协方差近似: rho * sqrt(s1_sq * s2_sq)
        cov12 = rho * sqrt(s1_sq * s2_sq);
        
        % 协方差矩阵 Sigma
        Sigma = [s1_sq, cov12; cov12, s2_sq];
        
        % 误差向量
        err = [data(1,i) - mu1; data(2,i) - mu2];
        
        % 累加二元正态对数似然
        % logL = logL - 0.5*log(det(Sigma)) - 0.5 * err' * inv(Sigma) * err
        try
            logL = logL - 0.5*log(det(Sigma)) - 0.5 * (err' / Sigma * err) - log(2*pi);
        catch
            logL = logL - 1e6; % 惩罚非正定矩阵
        end
    end
end

function [w1, w2] = jointResiduals(data, timeSteps, params, T)
    h = params(1); alpha = params(2); beta = params(3); rho = params(4);
    N = size(data, 2);
    w1 = zeros(1, N); w2 = zeros(1, N);
    
    for i = 2:N
        dt = timeSteps(i) - timeSteps(i-1);
        z1 = beta * (T - timeSteps(i-1))^(-alpha);
        z2 = beta * (T + 365 - timeSteps(i-1))^(-alpha);
        
        mu1 = xi_mean(data(1,i-1), h, z1, dt);
        mu2 = xi_mean(data(2,i-1), h, z2, dt);
        
        w1(i) = (data(1,i) - mu1) / sqrt(xi_var(z1, dt));
        w2(i) = (data(2,i) - mu2) / sqrt(xi_var(z2, dt));
    end
end

% 辅助数学工具函数
function m = xi_mean(xi_prev, h, z, dt)
    m = exp(0.5*dt*z)*xi_prev + h*dt*exp(0.5*dt*z)*sqrt(z);
end

function v = xi_var(z, dt)
    v = dt*z*exp(dt*z);
end
