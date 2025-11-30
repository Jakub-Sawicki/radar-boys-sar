function angleDetEstimated=MLE(Y)
    [M,N]=size(Y);
    R=Y*Y'/N;

    function J=CostFunction(angleDeg)
        aTemp=a(angleDeg,M);
        Pv=eye(M)-aTemp*(aTemp'*aTemp)^-1*aTemp';
        J=abs(trace(Pv*R));
    end
    Jsq=@(x)CostFunction(x);

   angleVec=[-45:0.1:45];
   pval=zeros(size(angleVec));
   for k=1:length(angleVec)
       pval(k)=Jsq(angleVec(k));
   end
   figure
   plot(angleVec,pval, 'LineWidth', 1.5);
    xlabel('Kąt [deg]');
    ylabel('Funkcja kosztu J(\theta)');
   grid on;
    hold off;

    % fminsearch can give incorrect results when starting point is in nearby local extremum
    angleDetEstimated=fminsearch(Jsq,0);
end
