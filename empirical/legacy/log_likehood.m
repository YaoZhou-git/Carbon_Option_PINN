clear all;clc;
if isempty(getenv('CARBON_OPTION_DATA_DIR')), error('Set CARBON_OPTION_DATA_DIR to a licensed legacy-data directory.'); end


A=xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'data202305-202404'),1,'B2:B260');
% 假设 A_i 是一个向量，包含了多个累积分布函数的值
A_i =A./100; % 累积分布函数值

% 使用 norminv 函数找到对应的 \xi_i 值
xi_i = norminv(A_i);



% 假设我们有一些观测数据和时间步长
data=xi_i;
timeSteps = 0:(length(A_i)-1);
T=length(A_i);


% 进行参数估计
[params, logL] = estimateParams(data, timeSteps, T);

% 输出结果
params
logL
[w]=residuals(data, timeSteps, params, T);
figure(1)
plot(A);
figure(2)
plot(w)
%% 经验自相关函数和QQ图
% 计算ACF
ACF_w=autocorr(w);

figure(3)
autocorr(w)
figure(4)
qqplot(w)


%% 计算residuals
function w = residuals(data, timeSteps, params, T)
    h = params(1);
    alpha = params(2);
    beta = params(3);
    
    logL = 0;
    for i = 2:length(data)
        t_i = timeSteps(i);
        t_i_minus_1 = timeSteps(i-1);
        xi_t_i = data(i);
        xi_t_i_minus_1 = data(i-1);
        
        % 计算z_t
        z_t_i_minus_1 = beta*(T - t_i_minus_1)^(-alpha);
        
        if z_t_i_minus_1 == 0
            % 处理 z_t_i_minus_1 为零的情况
            % 例如，可以设置一个很小的非零值
            z_t_i_minus_1 = 1e-10;
        end
        
        % 使用近似计算mu和sigma^2
        mu_i = exp(0.5*(t_i - t_i_minus_1)*z_t_i_minus_1)*xi_t_i_minus_1 + h*(t_i - t_i_minus_1)*exp(0.5*(t_i - t_i_minus_1)*z_t_i_minus_1)*sqrt(z_t_i_minus_1);
        sigma_i_squared = (t_i - t_i_minus_1)*z_t_i_minus_1*exp((t_i - t_i_minus_1)*z_t_i_minus_1);
        w_i=(xi_t_i-mu_i)/sqrt(sigma_i_squared);
        w(i)=w_i;
 
    end
end

%%


function [params, logL] = estimateParams(data, timeSteps, T)
    % 参数的初始猜测
    initParams = [0, 0.5, 0.5];
    
    % 目标函数（负对数似然）
    objFun = @(params) -logLikelihood(data, timeSteps, params, T);
    
    % 优化算法的选项
    options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', 'MaxIter', 10000);
    % options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
    % 执行优化
    [params, fval] = fminunc(objFun, initParams, options);
    
    % 最大对数似然
    logL = -fval;
end

function logL = logLikelihood(data, timeSteps, params, T)
    h = params(1);
    alpha = params(2);
    beta = params(3);
    
    logL = 0;
    for i = 2:length(data)
        t_i = timeSteps(i);
        t_i_minus_1 = timeSteps(i-1);
        xi_t_i = data(i);
        xi_t_i_minus_1 = data(i-1);
        
        % 计算z_t
        z_t_i_minus_1 = beta*(T - t_i_minus_1)^(-alpha);
        
        if z_t_i_minus_1 == 0
            % 处理 z_t_i_minus_1 为零的情况
            % 例如，可以设置一个很小的非零值
            z_t_i_minus_1 = 1e-10;
        end
        
        % 使用近似计算mu和sigma^2
        mu_i = exp(0.5*(t_i - t_i_minus_1)*z_t_i_minus_1)*xi_t_i_minus_1 + h*(t_i - t_i_minus_1)*exp(0.5*(t_i - t_i_minus_1)*z_t_i_minus_1)*sqrt(z_t_i_minus_1);
        sigma_i_squared = (t_i - t_i_minus_1)*z_t_i_minus_1*exp((t_i - t_i_minus_1)*z_t_i_minus_1);
        
        % 对数似然的累加
        logL = logL - ((xi_t_i - mu_i)^2)/(2*sigma_i_squared) - log(sqrt(2*pi*sigma_i_squared));
    end
end


