function out=a(PhiDeg,M)
    cfg=getConfig();
    cfg.lambda;
    k = 2*pi*cfg.d/cfg.lambda;
    out=exp(1j*[0:M-1]'*k*sin(PhiDeg*pi/180));
end
