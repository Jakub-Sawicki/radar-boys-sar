function out=a(PhiDeg,M)
    out=exp(j*[0:M-1]'*pi*sin(PhiDeg*pi/180));
end
