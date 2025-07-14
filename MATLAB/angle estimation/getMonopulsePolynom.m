function [monopulsePolynom, angleRange]=getMonopulsePolynom(M=getConfig().M)
    angleVec=[-90:0.01:90];

    deltaPattern=[-ones(1,M/2) ones(1,M/2)];

    SigmaBeam=a(0,M)'*a(angleVec,M);
    DeltaBeam=(a(0,M)'.*deltaPattern)*a(angleVec,M);

    % monopulse works only when angle is in sigma mainlobe region
    % looking for mainlobe bounds
    % bounded for -/+ 5 monopulse value
    [val,lowerIdx]=max(SigmaBeam);
    [val,upperIdx]=max(SigmaBeam);
    while imag(DeltaBeam(lowerIdx)/SigmaBeam(lowerIdx))>-5
        lowerIdx=lowerIdx-1;
    end
    while imag(DeltaBeam(upperIdx)/SigmaBeam(upperIdx))<5
        upperIdx=upperIdx+1;
    end

    angleRange=[angleVec(lowerIdx) angleVec(upperIdx)];

    monopulseAngles=angleVec(lowerIdx:upperIdx);
    monopulseValues=DeltaBeam(lowerIdx:upperIdx)./SigmaBeam(lowerIdx:upperIdx);

    % rank of polynom
    monopulsePolynom=polyfit(imag(monopulseValues), monopulseAngles,17);

%    figure
%    plot(angleVec,abs(SigmaBeam))
%    hold on
%    plot(angleVec,abs(DeltaBeam),'r')
%    legend('\Sigma beam','\Delta beam')

%    figure
%    plot(monopulseAngles,imag(monopulseValues))
%    hold on
%    plot(polyval(monopulsePolynom, imag(monopulseValues)), imag(monopulseValues),'.');
%    xlabel('Angle [deg]')
%    xlabel('MIMP value')
%    legend('monopulse curve','polynom fit')
end

