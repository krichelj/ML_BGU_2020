function w = softsvm(lambda, m, d, Xtrain, Ytrain)
 
  H = sparse(diag([2 * lambda * ones(1, d), zeros(1, m)]));
  f = sparse([zeros(1, d), (1/m) * ones(1, m)]');
  YiXi = zeros(m, d);
  
  for i=1:m
    YiXi(i, :) = Ytrain(i, :) * Xtrain(i, :);
  end
  
  Im = speye(m);
  A = sparse(-[zeros(m, d) full(Im);YiXi full(Im)]);
  b = sparse(-[zeros(1, m) ones(1, m)]');
  
  w = quadprog(H, f, A, b);
  
end