clear;
close all;

k  = 1.4;                        % Gamma (ratio of specific heats)
Ma = 0 : 0.00001 : 5.5;        % Mach number vector

exponent    = 1 / (k - 1);
rho_ratio   = 1 ./ (1 + ((k - 1) / 2) .* Ma.^2) .^ exponent;

figure('Name', 'Density Ratio vs Mach Number', 'NumberTitle', 'off');
plot(Ma, rho_ratio, 'g-', 'LineWidth', 1.5);
grid on;
xlabel('Mach Number (Ma)', 'FontSize', 12);
ylabel('\rho / \rho_R', 'FontSize', 12);
title(sprintf('Density Ratio  \\rho/\\rho_R  vs Mach Number  (\\gamma = %.2f)', k), ...
      'FontSize', 14);
xlim([0 5.5]);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

output_table = table(Ma', rho_ratio', 'VariableNames', {'Mach', 'rho_rho_R'});
writetable(output_table, 'density_ratio_results.csv');
