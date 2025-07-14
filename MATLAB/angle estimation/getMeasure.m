function out=getMeasure(targetDeg,ampl)
    cfg=getConfig();
    out=a(targetDeg,cfg.M)*ampl';
    out=out+crandn([cfg.M 1])/sqrt(2);
end
