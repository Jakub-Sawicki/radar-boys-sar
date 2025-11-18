function angleDetEstimated=BeamMVDR(Y)
    [M,N]=size(Y);
    R=Y*Y'/N;
    PMVDR=@(x)(-abs((a(x,M)'*R^-1*a(x,M))^-1/M));

   angleVec=[-45:0.1:45];
   pval=zeros(size(angleVec));
   for k=1:length(angleVec)
       pval(k)=PMVDR(angleVec(k));
   end
   figure
   plot(angleVec,pval);

    % fminsearch can give incorrect results when starting point is in nearby local extremum
    angleDetEstimated=fminsearch(PMVDR,0);
end
