function angleDetEstimated=MIMP(Y,polynom=getMonopulsePolynom(),M=getConfig().M)
    deltaPattern=[-ones(1,M/2) ones(1,M/2)];
    SigmaBeam=a(0,M)'*Y;
    DeltaBeam=(a(0,M)'.*deltaPattern)*Y;
    mimpValue=imag(DeltaBeam/SigmaBeam);

    angleDetEstimated=polyval(polynom,mimpValue);
end

