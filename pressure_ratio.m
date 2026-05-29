clc;
clear;
close all;

k  = 1.4;                        % Gamma (ratio of specific heats)

Ma = 0 : 0.00001 : 5.5;        % Mach number vector

exponent  = k / (k - 1);
p_ratio   = 1 ./ (1 + ((k - 1) / 2) .* Ma.^2) .^ exponent;


figure('Name', 'Pressure Ratio vs Mach Number', 'NumberTitle', 'off');
plot(Ma, p_ratio, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Mach Number (Ma)', 'FontSize', 12);
ylabel('p / p_R', 'FontSize', 12);
title(sprintf('Pressure Ratio  p/p_R  vs Mach Number  (\\gamma = %.2f)', k), ...
      'FontSize', 14);
xlim([0 5.5]);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

output_table = table(Ma', p_ratio', 'VariableNames', {'Mach', 'p_p_R'});
writetable(output_table, 'pressure_ratio_results.csv');
