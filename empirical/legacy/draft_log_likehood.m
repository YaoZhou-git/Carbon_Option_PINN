clear all;clc;
if isempty(getenv('CARBON_OPTION_DATA_DIR')), error('Set CARBON_OPTION_DATA_DIR to a licensed legacy-data directory.'); end
%
A=xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'数据.xlsx'),1,'D2:E339');
% 假设 A_i 是一个向量，包含了多个累积分布函数的值
A_i =A(:,2)./100; % 示例累积分布函数值

% 使用 norminv 函数找到对应的 \xi_i 值
xi_i = norminv(A_i);



% 假设我们有一些观测数据和时间步长
data=xi_i;
timeSteps=A(:,1);
T=337;
% data = [1, 1.2, 1.5, 2.1, 2.9]; % 例如，这是xi的观测值
% timeSteps = [0, 1, 2, 3, 4]; % 对应的时间步长
% T = 4; % 总时间



% 假设这是您想要估计的参数
params = [0, 0.5, 0.5]; % 初始猜测

% 蒙特卡洛模拟的迭代次数
nIterations = 10000;

% 存储似然函数值的数组
likelihoods = zeros(nIterations, 1);

% 进行蒙特卡洛模拟
for i = 1:nIterations
    % 生成随机样本
    randomData = generateRandomData(params(1), params(2), params(3));
    
    % 计算似然函数值
    likelihoods(i) = logLikelihood(randomData, timeSteps, params, T);
end

% 找到最大似然估计的参数值
[bestParams, bestLikelihood] = max(likelihoods);

% 输出结果
bestParams
bestLikelihood

function randomData = generateRandomData(h, alpha, beta,timeSteps)
    % 生成随机序列
    randomData = [0]; % 初始位置为0
    for t = 1:length(timeSteps) - 1
        randomData = [randomData; randomData(end) + h.*(timeSteps(t+1) - timeSteps(t)).*beta.*(T - timeSteps(t))];
    end
    
    % 添加噪声
    for t = 1:length(randomData)
        randomData(t) = randomData(t) + randn();
    end
end
% function randomData = generateRandomData(params,timeSteps)
%     % 假设参数是步长、初始位置和噪声水平
%     h = params(1);
%     alpha = params(2);
%     beta = params(3);
%     
%     % 生成随机序列
%     randomData = [0]; % 初始位置为0
%     for t = 1:length(timeSteps) - 1
%         randomData = [randomData; randomData(end) + h*(timeSteps(t+1) - timeSteps(t))*beta*(T - timeSteps(t))];
%     end
%     
%     % 添加噪声
%     for t = 1:length(randomData)
%         randomData(t) = randomData(t) + randn();
%     end
% end
% 
% function randomData = generateRandomData(params,timeSteps)
%     % 假设参数是步长、初始位置和噪声水平
%     h = params(1);
%     alpha = params(2);
%     beta = params(3);
%     
%     % 生成随机序列
%     randomData = [0]; % 初始位置为0
%     for t = 1:337
%         randomData = [randomData; randomData(end) + h*(timeSteps(t+1) - timeSteps(t))*beta*(T - timeSteps(t))];
%     end
%     
%     % 添加噪声
%     randomData = [randomData; randomData(end) + randn()];
% end

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
        mu_i = exp(0.5*(t_i - t_i_minus_1)*z_t_i_minus_1)*xi_t_i_minus_1 + h*(t_i - t_i_minus_1)*exp((t_i - t_i_minus_1)*z_t_i_minus_1)*sqrt(z_t_i_minus_1);
        sigma_i_squared = (t_i - t_i_minus_1)*z_t_i_minus_1*exp((t_i - t_i_minus_1)*z_t_i_minus_1);
        
        % 对数似然的累加
        logL = logL - ((xi_t_i - mu_i)^2)/(2*sigma_i_squared) - log(sqrt(2*pi*sigma_i_squared));
    end
end

