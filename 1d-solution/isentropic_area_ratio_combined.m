% =====================================================================
%  COMBINED 1-D ISENTROPIC FLOW PROGRAM
%  Area ratio  ->  Mach number (subsonic & supersonic)
%              ->  T/T_R, p/p_R, rho/rho_R  (temperature, pressure, density)
%
%  Equations used (taken ONLY from the original scripts):
%
%   (1) Area-Mach relation          (mach_subsonic / mach_supersonic_every_step.m)
%         A/A* = (1/Ma) * [1 + alpha*(Ma^2 - 1)]^n
%         n = (k+1)/(2(k-1)),  alpha = (k-1)/(k+1),  beta = 2(k-1)/(k+1)
%       solved for Ma with Newton-Raphson using
%         f (Ma) = (1/Ma)[1 + alpha(Ma^2-1)]^n - A/A*
%         f'(Ma) = -(1/Ma^2)[1 + alpha(Ma^2-1)]^n
%                  + (1/Ma) n [1 + alpha(Ma^2-1)]^(n-1) beta Ma
%
%   (2) Temperature ratio           (temperature_ratio.m)
%         T/T_R   = 1 / (1 + (k-1)/2 * Ma^2)
%   (3) Pressure ratio              (pressure_ratio.m)
%         p/p_R   = 1 / (1 + (k-1)/2 * Ma^2)^(k/(k-1))
%   (4) Density ratio               (density_ratio.m)
%         rho/rho_R = 1 / (1 + (k-1)/2 * Ma^2)^(1/(k-1))
%
%  Every A*/A value has two Mach roots, one subsonic and one supersonic.
%  The program finds both, then gets T/T_R, p/p_R, rho/rho_R for each root.
% =====================================================================

clc; clear; close all;

%% ---- USER INPUTS ---------------------------------------------------
k            = 1.4;       % ratio of specific heats
AstarA_start = 0.001;     % starting A*/A  (must be > 0 and <= 1)
AstarA_end   = 1.0;       % ending   A*/A  (= 1 is the throat)
AstarA_step  = 0.00001;   % step size of A*/A
tol          = 1e-9;      % Newton-Raphson convergence tolerance
max_iter     = 200;       % maximum Newton-Raphson iterations

% Optional single-point query: enter one or more A/A* values (>= 1)
% and the program prints Ma, T/T_R, p/p_R, rho/rho_R for both branches.
% Leave empty [] to skip.
AR_query     = [1.0 1.5 2.0 5.0 10.0 25.0];

output_csv   = 'combined_isentropic_results.csv';

%% ---- INPUT CHECKS --------------------------------------------------
if AstarA_start <= 0 || AstarA_end > 1
    error('A*/A must satisfy  0 < A*/A <= 1.');
end
if AstarA_start > AstarA_end
    error('AstarA_start must be <= AstarA_end.');
end
if any(AR_query < 1)
    error('Query values of A/A* must be >= 1.');
end

%% ---- CONSTANTS (from the area-Mach scripts) ------------------------
n     = (k + 1) / (2*(k - 1));
alpha = (k - 1) / (k + 1);
beta  = 2*(k - 1) / (k + 1);

f  = @(Ma, AR)  (1./Ma) .* (1 + alpha*(Ma.^2 - 1)).^n  -  AR;
fp = @(Ma)      -(1./Ma.^2).*(1 + alpha*(Ma.^2-1)).^n ...
                + (1./Ma).*n.*(1 + alpha*(Ma.^2-1)).^(n-1).*beta.*Ma;

%% ---- ISENTROPIC RATIOS (from the P, T, rho scripts) ----------------
T_ratio   = @(Ma) 1 ./ (1 + ((k - 1) / 2) .* Ma.^2);
p_ratio   = @(Ma) 1 ./ (1 + ((k - 1) / 2) .* Ma.^2) .^ (k / (k - 1));
rho_ratio = @(Ma) 1 ./ (1 + ((k - 1) / 2) .* Ma.^2) .^ (1 / (k - 1));

%% ---- AREA RATIO VECTOR ---------------------------------------------
AstarA_vec = (AstarA_start : AstarA_step : AstarA_end)';
if abs(AstarA_vec(end) - AstarA_end) > 1e-12      % make sure throat is included
    AstarA_vec(end+1) = AstarA_end;
end
AR_vec = 1 ./ AstarA_vec;                           % A/A* = 1/(A*/A)
N      = numel(AstarA_vec);
fprintf('Solving %d area-ratio points (A*/A = %g ... %g, step %g)\n', ...
        N, AstarA_start, AstarA_end, AstarA_step);

%% ---- SUBSONIC BRANCH  (Ma < 1) -------------------------------------
Ma_guess_sub = min(0.95, AstarA_vec);               % same guess as subsonic script
[Ma_sub, it_sub, res_sub] = newton_area_mach(Ma_guess_sub, AR_vec, f, fp, ...
                                             tol, max_iter, 1e-8);

%% ---- SUPERSONIC BRANCH (Ma > 1) ------------------------------------
Ma_guess_sup = 1.0 + (AR_vec - 1)*0.9 + 0.05;       % same guess as supersonic script
[Ma_sup, it_sup, res_sup] = newton_area_mach(Ma_guess_sup, AR_vec, f, fp, ...
                                             tol, max_iter, 1.0 + 1e-9);

%% ---- PROPERTY RATIOS ----------------------------------------------
T_sub   = T_ratio(Ma_sub);    T_sup   = T_ratio(Ma_sup);
p_sub   = p_ratio(Ma_sub);    p_sup   = p_ratio(Ma_sup);
rho_sub = rho_ratio(Ma_sub);  rho_sup = rho_ratio(Ma_sup);

fprintf('Subsonic  : max residual = %.3e, max iterations = %d\n', max(res_sub), max(it_sub));
fprintf('Supersonic: max residual = %.3e, max iterations = %d\n', max(res_sup), max(it_sup));

%% ---- SAVE COMBINED TABLE ------------------------------------------
Results = table(AstarA_vec, AR_vec, ...
                Ma_sub, T_sub, p_sub, rho_sub, it_sub, res_sub, ...
                Ma_sup, T_sup, p_sup, rho_sup, it_sup, res_sup, ...
    'VariableNames', {'AstarA','AR', ...
        'Ma_subsonic','T_T_R_sub','p_p_R_sub','rho_rho_R_sub','Iter_sub','Residual_sub', ...
        'Ma_supersonic','T_T_R_sup','p_p_R_sup','rho_rho_R_sup','Iter_sup','Residual_sup'});
writetable(Results, output_csv);
fprintf('Results written to %s\n\n', output_csv);

%% ---- SINGLE-POINT QUERY -------------------------------------------
if ~isempty(AR_query)
    ARq = AR_query(:);
    Mq_sub = newton_area_mach(min(0.95, 1./ARq), ARq, f, fp, tol, max_iter, 1e-8);
    Mq_sup = newton_area_mach(1.0 + (ARq - 1)*0.9 + 0.05, ARq, f, fp, tol, max_iter, 1.0 + 1e-9);

    fprintf('%8s | %10s %9s %9s %10s | %10s %9s %9s %10s\n', 'A/A*', ...
            'Ma_sub','T/T_R','p/p_R','rho/rho_R','Ma_sup','T/T_R','p/p_R','rho/rho_R');
    fprintf('%s\n', repmat('-', 1, 96));
    for i = 1:numel(ARq)
        fprintf('%8.4f | %10.6f %9.6f %9.6f %10.6f | %10.6f %9.6f %9.6f %10.6f\n', ARq(i), ...
            Mq_sub(i), T_ratio(Mq_sub(i)), p_ratio(Mq_sub(i)), rho_ratio(Mq_sub(i)), ...
            Mq_sup(i), T_ratio(Mq_sup(i)), p_ratio(Mq_sup(i)), rho_ratio(Mq_sup(i)));
    end
    fprintf('\n');
end

%% ---- PLOTS ---------------------------------------------------------
% 1) Mach number vs area ratio (both branches)
figure('Name', 'Mach Number vs Area Ratio', 'NumberTitle', 'off');
semilogx(AR_vec, Ma_sub, 'b-', AR_vec, Ma_sup, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Area ratio  A/A^*', 'FontSize', 12);
ylabel('Mach Number (Ma)', 'FontSize', 12);
legend('Subsonic branch', 'Supersonic branch', 'Location', 'northwest');
title(sprintf('Mach Number vs Area Ratio  (\\gamma = %.2f)', k), 'FontSize', 14);
set(gca, 'FontSize', 11);

% 2) T, p, rho ratios vs area ratio, subsonic branch
figure('Name', 'Property Ratios vs Area Ratio (Subsonic)', 'NumberTitle', 'off');
semilogx(AR_vec, T_sub, 'b-', AR_vec, p_sub, 'r-', AR_vec, rho_sub, 'g-', 'LineWidth', 1.5);
grid on;
xlabel('Area ratio  A/A^*', 'FontSize', 12);
ylabel('Ratio to reservoir value', 'FontSize', 12);
legend('T/T_R', 'p/p_R', '\rho/\rho_R', 'Location', 'southeast');
title(sprintf('Subsonic Branch: T, p, \\rho Ratios vs A/A^*  (\\gamma = %.2f)', k), 'FontSize', 14);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

% 3) T, p, rho ratios vs area ratio, supersonic branch
figure('Name', 'Property Ratios vs Area Ratio (Supersonic)', 'NumberTitle', 'off');
semilogx(AR_vec, T_sup, 'b-', AR_vec, p_sup, 'r-', AR_vec, rho_sup, 'g-', 'LineWidth', 1.5);
grid on;
xlabel('Area ratio  A/A^*', 'FontSize', 12);
ylabel('Ratio to reservoir value', 'FontSize', 12);
legend('T/T_R', 'p/p_R', '\rho/\rho_R', 'Location', 'northeast');
title(sprintf('Supersonic Branch: T, p, \\rho Ratios vs A/A^*  (\\gamma = %.2f)', k), 'FontSize', 14);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

% 4) T, p, rho ratios vs Mach number (full range covered by both branches)
Ma_all  = [flipud(Ma_sub); Ma_sup];
figure('Name', 'Property Ratios vs Mach Number', 'NumberTitle', 'off');
plot(Ma_all, T_ratio(Ma_all), 'b-', Ma_all, p_ratio(Ma_all), 'r-', ...
     Ma_all, rho_ratio(Ma_all), 'g-', 'LineWidth', 1.5);
grid on;
xlabel('Mach Number (Ma)', 'FontSize', 12);
ylabel('Ratio to reservoir value', 'FontSize', 12);
legend('T/T_R', 'p/p_R', '\rho/\rho_R', 'Location', 'northeast');
title(sprintf('T, p, \\rho Ratios vs Mach Number  (\\gamma = %.2f)', k), 'FontSize', 14);
ylim([0 1.05]);
set(gca, 'FontSize', 11);

% 5) Area ratio vs Mach number (the usual nozzle curve)
figure('Name', 'Area Ratio vs Mach Number', 'NumberTitle', 'off');
semilogy(Ma_sub, AR_vec, 'b-', Ma_sup, AR_vec, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Mach Number (Ma)', 'FontSize', 12);
ylabel('Area ratio  A/A^*', 'FontSize', 12);
legend('Subsonic branch', 'Supersonic branch', 'Location', 'north');
title(sprintf('Area Ratio vs Mach Number  (\\gamma = %.2f)', k), 'FontSize', 14);
set(gca, 'FontSize', 11);


%% =====================================================================
%  LOCAL FUNCTION: Newton-Raphson on the area-Mach relation
%  It uses the same iteration and tolerance as the original scripts,
%  and the same lower limit on Ma.
%  It just runs on all points at once instead of one by one.
% =====================================================================
function [Ma, iters, residual] = newton_area_mach(Ma0, AR, f, fp, tol, max_iter, Ma_min)
    Ma       = Ma0(:);
    AR       = AR(:);
    iters    = zeros(size(Ma));
    active   = true(size(Ma));

    throat         = abs(AR - 1.0) < 1e-12;    % exact throat -> Ma = 1
    Ma(throat)     = 1.0;
    active(throat) = false;

    for it = 1:max_iter
        idx = find(active);
        if isempty(idx), break; end

        fv  = f(Ma(idx), AR(idx));
        fpv = fp(Ma(idx));

        flat = abs(fpv) < 1e-15;               % derivative too small -> stop
        active(idx(flat)) = false;
        idx = idx(~flat);  fv = fv(~flat);  fpv = fpv(~flat);

        dMa     = fv ./ fpv;
        Ma(idx) = max(Ma(idx) - dMa, Ma_min);
        iters(idx) = it;

        active(idx(abs(dMa) < tol)) = false;   % converged
    end

    residual         = abs(f(Ma, AR));
    residual(throat) = 0;
end
