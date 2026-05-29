clc; clear; close all;

k            = 1.4;      % ratio of specific heats
AstarA_start = 0.001;      % starting A*/A  (must be > 0 and <= 1)
AstarA_end   = 1.0;      % ending   A*/A  (= 1 is the throat)
AstarA_step  = 0.00001;   % step size — every value printed
tol          = 1e-9;     % Newton-Raphson convergence tolerance
max_iter     = 200;      % maximum iterations

if AstarA_start <= 0 || AstarA_end > 1
    error('A*/A must satisfy  0 < A*/A <= 1.');
end
if AstarA_start > AstarA_end
    error('AstarA_start must be <= AstarA_end.');
end

AstarA_vec = AstarA_start : AstarA_step : AstarA_end;
AR_vec     = 1 ./ AstarA_vec;        % A/A* = 1/(A*/A)
N          = length(AstarA_vec);

n     = (k + 1) / (2*(k - 1));
alpha = (k - 1) / (k + 1);
beta  = 2*(k - 1) / (k + 1);

f  = @(Ma, AR)  (1./Ma) .* (1 + alpha*(Ma.^2 - 1)).^n  -  AR;
fp = @(Ma)      -(1./Ma.^2).*(1 + alpha*(Ma.^2-1)).^n ...
                + (1./Ma).*n.*(1 + alpha*(Ma.^2-1)).^(n-1).*beta.*Ma;

Ma_sup   = zeros(1, N);
iters    = zeros(1, N);
residual = zeros(1, N);

for i = 1:N
    AR = AR_vec(i);

    if abs(AR - 1.0) < 1e-12          % exact throat
        Ma_sup(i) = 1.0;  iters(i) = 0;  residual(i) = 0;
        continue
    end

    Ma = 1.0 + (AR - 1)*0.9 + 0.05;  % supersonic initial guess

    for it = 1:max_iter
        fv  = f(Ma, AR);
        fpv = fp(Ma);
        if abs(fpv) < 1e-15,  break;  end
        dMa = fv / fpv;
        Ma  = Ma - dMa;
        Ma  = max(Ma, 1.0 + 1e-9);    % keep above 1
        if abs(dMa) < tol,  break;  end
    end

    Ma_sup(i)   = Ma;
    iters(i)    = it;
    residual(i) = abs(f(Ma, AR));
end

T = table(AstarA_vec', AR_vec', Ma_sup', iters', residual', 'VariableNames', {'AstarA','AR','Ma_supersonic','Iterations','Residual'});
writetable(T, 'supersonic_every_step.csv');
