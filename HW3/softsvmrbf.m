function alpha = softsvmrbf(lambda, sigma, m, d, Xtrain, Ytrain)
  
  Gup = zeros(m);
  for i=1:m
    for j=i:m
      Gup(i,j) = exp(-((norm(Xtrain(i,:)-Xtrain(j,:)))^2 /(2*sigma)));
    end
  end
  
  Glo = Gup';
  Glo(1:m+1:m^2) = 0;
  G = Gup+Glo;
  
  epsilon = 0.000001;
  H = sparse(2 * lambda * blkdiag(G, zeros(m,m)));
  
  if (min(eig(H))<=0)
       H = H + eye(size(H))*epsilon;
  end
  
  f = sparse([zeros(1,m), (1/m) * ones(1, m)]');
  YiGi = (G'*diag(Ytrain))';
  
  Im = eye(m);
  A = sparse(-[zeros(m,m) Im; YiGi Im]);
  b = sparse(-[zeros(1, m) ones(1, m)]');
  
  alpha_t = quadprog(H,f,A,b);
  
  alpha = alpha_t(1:m,:);
end
