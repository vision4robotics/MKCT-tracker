function enhanced_res = REO(res, base)

% Resolution Enhancement Operator for Response
sum0 = sumsqr(res);
R = base .^ (res.^2);
e = R ./ sum(R(:));
enhanced_res = sqrt(e * sum0);

end