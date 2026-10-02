clc;
clear;
close all;

k  = 1.4;                        % Gamma (ratio of specific heats)

Ma = 0 : 0.00001 : 5.5;        % Mach number vector

T_ratio = 1 ./ (1 + ((k - 1) / 2) .* Ma.^2);

figure('Name', 'Temperature Ratio vs Mach Number', 'NumberTitle', 'off');
plot(Ma, T_ratio, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Mach Number (Ma)', 'FontSize', 12);
ylabel('T / T_R', 'FontSize', 12);
title(sprintf('Temperature Ratio  T/T_R  vs Mach Number  (\\gamma = %.2f)', k), ...
      'FontSize', 14);
xlim([0 5.5]);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

output_table = table(Ma', T_ratio', 'VariableNames', {'Mach', 'T_T_R'});
writetable(output_table, 'temperature_ratio_results.csv');
