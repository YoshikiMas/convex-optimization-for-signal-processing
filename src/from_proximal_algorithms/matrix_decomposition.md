# Matrix decomposition example in Proximal algorithms


Unofficial implementation of an example of matrix decomposition in proximal algorithms [1]. 



```matlab:Code
clear all;
rng(0);

% Data size
m = 20;  % number of rows
n = 50;  % number of colmuns

% Params
MAX_ITER = 300;
ABSTOL   = 1e-6;
RELTOL   = 1e-6;
N = 3;  % number of components
r = 4;  % rank
gamma_ratio = 0.15;
```


```matlab:Code
%  Data preparation
L = randn(m,r) * randn(r,n);
S = sprandn(m,n,0.05);
S(S ~= 0) = 20*binornd(1,0.5,nnz(S),1)-10;
V = 0.01*randn(m,n);

A = S + L + V;
```


```matlab:Code
% Set gamma
g2_max = norm(A(:),inf);
g3_max = norm(A);
g2 = gamma_ratio*g2_max;
g3 = gamma_ratio*g3_max;
```


```matlab:Code
% ADMM
X_1 = zeros(m,n);
X_2 = zeros(m,n);
X_3 = zeros(m,n);
z   = zeros(m,N*n);
U   = zeros(m,n);
lambda = 1;
rho = 1/lambda;

tic;
fprintf('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
```


```text:Output
iter	    r norm	   eps pri	    s norm	  eps dual	 objective
```


```matlab:Code

for k = 1:MAX_ITER

    B = avg(X_1, X_2, X_3) - A./N + U;

    % x-update
    X_1 = (1/(1+lambda))*(X_1 - B);
    X_2 = prox_l1(X_2 - B, lambda*g2);
    X_3 = prox_matrix(X_3 - B, lambda*g3, @prox_l1);

    % (for termination checks only)
    x = [X_1 X_2 X_3];
    zold = z;
    z = x + repmat(-avg(X_1, X_2, X_3) + A./N, 1, N);

    % u-update
    U = B;

    % diagnostics, reporting, termination checks
    h.objval(k)   = objective(X_1, g2, X_2, g3, X_3);
    h.r_norm(k)   = norm(x - z,'fro');
    h.s_norm(k)   = norm(-rho*(z - zold),'fro');
    h.eps_pri(k)  = sqrt(m*n*N)*ABSTOL + RELTOL*max(norm(x,'fro'), norm(-z,'fro'));
    h.eps_dual(k) = sqrt(m*n*N)*ABSTOL + RELTOL*sqrt(N)*norm(rho*U,'fro');

    if k == 1 || mod(k,10) == 0
        fprintf('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            h.r_norm(k), h.eps_pri(k), h.s_norm(k), h.eps_dual(k), h.objval(k));
    end

    if h.r_norm(k) < h.eps_pri(k) && h.s_norm(k) < h.eps_dual(k)
         break;
    end

end
```


```text:Output
   1	   41.7766	    0.0001	   61.5891	    0.0001	    433.51
  10	    9.7749	    0.0001	    7.9580	    0.0001	   2105.38
  20	    3.9687	    0.0001	    2.3887	    0.0001	   2217.10
  30	    1.2071	    0.0001	    0.6616	    0.0001	   2131.29
  40	    0.4276	    0.0001	    0.2262	    0.0001	   2124.76
  50	    0.1036	    0.0001	    0.0606	    0.0001	   2122.78
  60	    0.0313	    0.0001	    0.0200	    0.0001	   2123.38
  70	    0.0133	    0.0001	    0.0058	    0.0001	   2123.56
  80	    0.0042	    0.0001	    0.0024	    0.0001	   2123.34
  90	    0.0013	    0.0001	    0.0007	    0.0001	   2123.38
 100	    0.0003	    0.0001	    0.0002	    0.0001	   2123.39
 110	    0.0001	    0.0001	    0.0001	    0.0001	   2123.39
```


```matlab:Code

h.admm_toc = toc;
h.admm_iter = k;
h.X1_admm = X_1;
h.X2_admm = X_2;
h.X3_admm = X_3;

fprintf('\nADMM (vs true):\n');
```


```text:Output
ADMM (vs true):
```


```matlab:Code
fprintf('|V| = %.2f;  |X_1| = %.2f\n', norm(V, 'fro'), norm(X_1,'fro'));
```


```text:Output
|V| = 0.31;  |X_1| = 26.23
```


```matlab:Code
fprintf('nnz(S) = %d; nnz(X_2) = %d\n', nnz(S), nnz(X_2));
```


```text:Output
nnz(S) = 49; nnz(X_2) = 53
```


```matlab:Code
fprintf('rank(L) = %d; rank(X_3) = %d\n', rank(L), rank(X_3));
```


```text:Output
rank(L) = 4; rank(X_3) = 4
```


```matlab:Code
function p = objective(X_1, g2, X_2, g3, X_3)
    d = svd(X_3,'econ');
    p = 0.5*norm(X_1,'fro')^2 + g2*norm(X_2(:),1) + g3*norm(d,1);
end
function x = avg(varargin)
    N = length(varargin);
    x = 0;
    for k = 1:N
        x = x + varargin{k};
    end
    x = x/N;
end

function x = prox_l1(v, lambda)
    x = max(0, v - lambda) - max(0, -v - lambda);
end

function x = prox_matrix(v, lambda, prox_f)
    [U,S,V] = svd(v,'econ');
    x = U*diag(prox_f(diag(S), lambda))*V';
end
```

## Reference


[1] N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and Trends in Optimization, 1(3):123-231, 2014.




Official Code: \href{https://web.stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html}{https://web.stanford.edu/\textasciitilde{}boyd/papers/prox_algs/matrix_decomp.html}




Proximal operators: [https://github.com/cvxgrp/proximal](https://github.com/cvxgrp/proximal)


