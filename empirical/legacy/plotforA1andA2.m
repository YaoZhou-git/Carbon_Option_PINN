%% 1. 数据读取与转换
if isempty(getenv('CARBON_OPTION_DATA_DIR')), error('Set CARBON_OPTION_DATA_DIR to a licensed legacy-data directory.'); end
% 读取价格数据
A1_raw = xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'EUA Future Dec24 and Dec25 price in 24.xlsx'), 1, 'B3:B249'); 
A2_raw = xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'EUA Future Dec24 and Dec25 price in 24.xlsx'), 1, 'D3:D249');

% 读取日期字符串 (从 raw 或 txt 中提取)
[~, ~, raw] = xlsread(fullfile(getenv('CARBON_OPTION_DATA_DIR'),'EUA Future Dec24 and Dec25 price in 24.xlsx'), 1, 'A3:A249');

% 转换日期：将 cell 数组转换为 datetime 对象
% 'InputFormat' 必须与你 Excel 里的字符串格式匹配，如 'yyyy/MM/dd' 或 'yyyy/M/d'
t = datetime(raw, 'InputFormat', 'yyyy/MM/dd');

%% 2. 绘制图形
figure('Color', 'w', 'Position', [100, 100, 900, 500]); 
hold on;

% 绘制两条曲线，增加颜色区分
p1 = plot(t, A1_raw, 'Color', [0 0.4470 0.7410], 'LineWidth', 1.8, 'DisplayName', 'EUA Dec 2024 (A_1)');
p2 = plot(t, A2_raw, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.8, 'DisplayName', 'EUA Dec 2025 (A_2)');

%% 3. X 轴精细化设置
% 设置 X 轴限制
xlim([min(t) max(t)]);

% 设置刻度间隔为 1 个月
% 'TicksPerLabel' 可以控制显示密度
ax = gca;
ax.XAxis.TickLabelFormat = 'yyyy/MM'; 
xticks(min(t):calmonths(1):max(t)); 

% 旋转并美化
xtickangle(45);
set(ax, 'TickDir', 'in', 'Box', 'on'); % 刻度线向外，去掉上/右边框

%% 4. 细节增强
grid off;
set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.3);
xlabel('Observation Date (Monthly)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Settlement Price (EUR/tonne)',  'FontSize', 14,'FontWeight', 'bold');
% title('Structural Comparison: EUA Dec 2024 vs. Dec 2025 Futures', 'FontSize', 14);
ax.XAxis.FontWeight = 'bold';
ax.YAxis.FontWeight = 'bold';
legend('Location', 'best', 'FontSize', 14,'FontWeight','bold');

% 为了更直观，可以在下方显示价差 Spread 的数值规律
% text(t(10), max(A1_raw), ['Avg Spread: ' num2str(mean(A2_raw - A1_raw), '%.2f')], 'FontSize', 10);

hold off;
